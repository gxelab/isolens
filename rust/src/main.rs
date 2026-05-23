use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, BufRead, BufReader, Write},
    path::Path,
};

use flate2::write::GzEncoder;
use flate2::Compression;
use lz4_flex::frame::FrameDecoder as Lz4Decoder;
use noodles_sam::alignment::{
    record::{
        cigar::{op::Kind, Op},
        data::field::{value::Array, Tag, Value},
        Cigar,
    },
};
// Bring in the unified alignment toolkit
use noodles_util::alignment;

#[derive(Debug, Clone)]
struct TargetAssignment {
    tx_id: u32,
    prob: f32,
}

#[derive(Default, Clone)]
struct PositionStats {
    n_read: usize,
    sum_probs: f32,
    n_nomod: usize,
    wt_nomod: f32,
    mods: HashMap<String, (usize, f32), ahash::RandomState>,
}

#[derive(Default, Clone)]
struct PolyAStats {
    n_reads: usize,
    sum_weights: f32,
    pa_reads: usize,
    pa_weights: f32,
    sum_weighted_pa_len: f32,
    probs: Vec<f32>,
    pa_lens: Vec<i32>,
}

fn print_help() {
    println!(
        r#"isolens: High-performance transcript-level base modification and Poly(A) tail profiling pipeline

USAGE:
    isolens -b/--bam <PATH> -p/--prob <PATH> --out1 <PATH> --out2 <PATH> [OPTIONS]

OPTIONS:
    -b, --bam <PATH>         Path to input BAM alignment file (Required)
    -p, --prob <PATH>        Path to oarfish assignment probability map (.lz4) (Required)
    --out1 <PATH>            Output TSV path for positional modification summary (Required)
    --out2 <PATH>            Output TSV path for poly(A) statistics summary (Required)
    --out-per-base <PATH>    Optional path to write raw per-read modification lines
    -t, --threshold <FLOAT>  Modification probability threshold [default: 0.95]
    -j, --threads <INT>      Number of decompression background worker threads [default: 4]
    -z, --gzip               Compress outputs using gzip
    -v, --verbose            Show alignment iteration progression and diagnostics
    -h, --help               Print this help manual
"#
    );
}

fn create_writer(path: &str, use_gzip: bool) -> io::Result<Box<dyn Write>> {
    let file = File::create(path)?;
    if use_gzip {
        Ok(Box::new(GzEncoder::new(file, Compression::default())))
    } else {
        Ok(Box::new(io::BufWriter::with_capacity(128 * 1024, file)))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args() {
        Ok(Some(a)) => a,
        Ok(None) => return Ok(()),
        Err(e) => {
            eprintln!("Argument Error: {e}");
            std::process::exit(1);
        }
    };

    let ml_threshold_u8 = (args.mod_threshold * 255.0).round() as u8;

    if args.verbose {
        eprintln!("[isolens] Loading Oarfish allocations...");
    }
    let (tx_names, probabilities, name_to_id) = parse_oarfish(&args.oarfish_path)?;

    if args.verbose {
        eprintln!("[isolens] Spawning Unified Multi-Threaded Reader ({} cores)...", args.threads);
    }

    // Setup the unified alignment reader builder with explicit thread count
    let mut reader = alignment::io::reader::Builder::default()
        .build_from_path(&args.bam_path)?;

    let header = reader.read_header()?;
    let ref_seqs = header.reference_sequences();

    // Map reference sequence positions directly to internal transcript IDs
    let mut bam_ref_to_tx_id: Vec<Option<u32>> = vec![None; ref_seqs.len()];
    for (idx, (name, _)) in ref_seqs.iter().enumerate() {
        let name_str = std::str::from_utf8(name).unwrap_or("");
        if let Some(&id) = name_to_id.get(name_str) {
            bam_ref_to_tx_id[idx] = Some(id);
        }
    }

    let mut modification_summary: HashMap<(u32, usize), PositionStats, ahash::RandomState> =
        HashMap::with_hasher(ahash::RandomState::new());
    let mut poly_a_summary: HashMap<u32, PolyAStats, ahash::RandomState> =
        HashMap::with_hasher(ahash::RandomState::new());
    let mut seen_mod_types: HashSet<String, ahash::RandomState> =
        HashSet::with_hasher(ahash::RandomState::new());

    let mut per_base_writer: Option<Box<dyn Write>> = match &args.out_per_base {
        Some(path) => {
            let mut writer = create_writer(path, args.gzip)?;
            writeln!(writer, "read_id\tpos_read\tmod_type\tx_name\tprob\tpos_tx\tmod_prob")?;
            Some(writer)
        }
        None => None,
    };

    let tag_mm_upper = Tag::from([b'M', b'M']);
    let tag_mm_lower = Tag::from([b'M', b'm']);
    let tag_ml_upper = Tag::from([b'M', b'L']);
    let tag_ml_lower = Tag::from([b'M', b'l']);
    let tag_pt = Tag::from([b'p', b't']);

    let mut total_records = 0;

    // Option 3 uses unified record streaming which takes a reference to the header
    for result in reader.records(&header) {
        let record = result?;
        total_records += 1;

        if args.verbose && total_records % 1_000_000 == 0 {
            eprintln!("[isolens] Audited {} alignments...", total_records);
        }

        if record.flags()?.is_unmapped() { continue; }

        let read_id_bytes = match record.name() {
            Some(name) => name,
            None => continue,
        };
        let read_id_str = std::str::from_utf8(read_id_bytes).unwrap_or("");

        let assignments = match probabilities.get(read_id_str) {
            Some(a) => a,
            None => continue,
        };

        let tx_index = match record.reference_sequence_id(&header) {
            Some(idx) => idx?,
            None => continue,
        };

        let tx_id = match bam_ref_to_tx_id.get(tx_index).and_then(|&x| x) {
            Some(id) => id,
            None => continue,
        };

        let assignment = match assignments.iter().find(|a| a.tx_id == tx_id) {
            Some(a) => a,
            None => continue,
        };

        // --- Core Process 1: Track Poly(A) Tail Features ---
        // Bind the data block to a variable so its lifespan covers all tag lookups
        let data = record.data();

        let pt_value = data.get(&tag_pt);
        let mut pa_length = -1;
        if let Some(Ok(val)) = pt_value {
            if let Some(int_val) = val.as_int() {
                    pa_length = int_val as i32;
            }
        }

        let pa_entry = poly_a_summary.entry(tx_id).or_default();
        pa_entry.n_reads += 1;
        pa_entry.sum_weights += assignment.prob;
        pa_entry.probs.push(assignment.prob);
        pa_entry.pa_lens.push(pa_length);
        if pa_length > 0 {
            pa_entry.pa_reads += 1;
            pa_entry.pa_weights += assignment.prob;
            pa_entry.sum_weighted_pa_len += (pa_length as f32) * assignment.prob;
        }

        // --- Core Process 2: Parse Alignments and Modification Spaces ---
        // We reuse the 'data' binding we created above, avoiding the lifetime drop
        let mm_tag = data.get(&tag_mm_upper).or_else(|| data.get(&tag_mm_lower));
        let ml_tag = data.get(&tag_ml_upper).or_else(|| data.get(&tag_ml_lower));

        let cigar = record.cigar();
        let read_to_tx_map = map_read_to_tx(&cigar, record.alignment_start().unwrap()?.get());

        for &_maybe_tx_pos_1 in read_to_tx_map.iter() {
            if let Some(tx_pos_1) = _maybe_tx_pos_1 {
                let coord_stats = modification_summary.entry((tx_id, tx_pos_1)).or_default();
                coord_stats.n_read += 1;
                coord_stats.sum_probs += assignment.prob;
            }
        }

        if let Some(Ok(mm_value)) = mm_tag {
            let mm_str = match mm_value {
                Value::String(bstr) => std::str::from_utf8(bstr).unwrap_or(""),
                _ => continue,
            };

            let ml_bytes: Option<Vec<u8>> = if let Some(Ok(ml_val)) = ml_tag {
                match ml_val {
                    Value::Array(arr) => {
                        match &arr {
                            Array::Int8(v) => v.iter().map(|r| r.map(|val| val as u8)).collect::<Result<Vec<_>,_>>().ok(),
                            Array::UInt8(v) => v.iter().map(|r| r.map(|val| val as u8)).collect::<Result<Vec<_>,_>>().ok(),
                            Array::Int16(v) => v.iter().map(|r| r.map(|val| val as u8)).collect::<Result<Vec<_>,_>>().ok(),
                            Array::UInt16(v) => v.iter().map(|r| r.map(|val| val as u8)).collect::<Result<Vec<_>,_>>().ok(),
                            Array::Int32(v) => v.iter().map(|r| r.map(|val| val as u8)).collect::<Result<Vec<_>,_>>().ok(),
                            Array::UInt32(v) => v.iter().map(|r| r.map(|val| val as u8)).collect::<Result<Vec<_>,_>>().ok(),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            } else {
                None
            };

            let seq: Vec<u8> = record.sequence().iter().collect();
            let mut total_mod_instance_idx = 0;
            let mut modified_positions_in_read = HashMap::new();

            for mod_group in mm_str.split(';') {
                if mod_group.is_empty() { continue; }
                let parts: Vec<&str> = mod_group.split(',').collect();
                if parts.is_empty() { continue; }

                let meta = parts[0];
                if meta.len() < 3 { continue; }
                let target_base = meta.chars().next().unwrap() as u8;
                let mod_type = meta[2..].trim_end_matches('.').to_string();
                seen_mod_types.insert(mod_type.clone());

                let skips: Vec<usize> = parts[1..].iter().filter_map(|s| s.parse::<usize>().ok()).collect();
                let mut skip_idx = 0;
                let mut current_skip = skips.get(skip_idx).cloned();
                let mut occurrences_found = 0;

                for read_pos_0 in 0..seq.len() {
                    if seq[read_pos_0] == target_base {
                        if let Some(skip) = current_skip {
                            if occurrences_found == skip {
                                let mut passes_cutoff = true;
                                let mut base_prob = 1.0;

                                if let Some(ref prob_array) = ml_bytes {
                                    if let Some(&raw_prob) = prob_array.get(total_mod_instance_idx) {
                                        if raw_prob < ml_threshold_u8 {
                                            passes_cutoff = false;
                                        }
                                        base_prob = raw_prob as f32 / 255.0;
                                    }
                                }

                                if let Some(tx_pos_1) = read_to_tx_map[read_pos_0] {
                                    if passes_cutoff {
                                        modified_positions_in_read.insert((tx_pos_1, mod_type.clone()), assignment.prob);
                                    }

                                    if let Some(ref mut p_writer) = per_base_writer {
                                        writeln!(
                                            p_writer,
                                            "{}\t{}\t{}\t{}\t{:.4}\t{}\t{:.4}",
                                            read_id_str, read_pos_0 + 1, mod_type, tx_names[tx_id as usize], assignment.prob, tx_pos_1, base_prob
                                        )?;
                                    }
                                }

                                total_mod_instance_idx += 1;
                                skip_idx += 1;
                                current_skip = skips.get(skip_idx).cloned();
                                occurrences_found = 0;
                            } else {
                                occurrences_found += 1;
                            }
                        }
                    }
                }
            }

            for &_maybe_tx_pos_1 in read_to_tx_map.iter() {
                if let Some(tx_pos_1) = _maybe_tx_pos_1 {
                    let coord_stats = modification_summary.entry((tx_id, tx_pos_1)).or_default();
                    let mut matched_any_mod = false;
                    for m_type in &seen_mod_types {
                        if let Some(&read_weight) = modified_positions_in_read.get(&(tx_pos_1, m_type.clone())) {
                            let mod_entry = coord_stats.mods.entry(m_type.clone()).or_insert((0, 0.0));
                            mod_entry.0 += 1;
                            mod_entry.1 += read_weight;
                            matched_any_mod = true;
                        }
                    }

                    if !matched_any_mod {
                        coord_stats.n_nomod += 1;
                        coord_stats.wt_nomod += assignment.prob;
                    }
                }
            }
        } else {
            for &_maybe_tx_pos_1 in read_to_tx_map.iter() {
                if let Some(tx_pos_1) = _maybe_tx_pos_1 {
                    let coord_stats = modification_summary.entry((tx_id, tx_pos_1)).or_default();
                    coord_stats.n_nomod += 1;
                    coord_stats.wt_nomod += assignment.prob;
                }
            }
        }
    }

    if let Some(mut p_writer) = per_base_writer {
        p_writer.flush()?;
    }

    // --- Output Generation 1: Positional Summaries ---
    if args.verbose {
        eprintln!("[isolens] Writing Output 1 (Positional Summaries)...");
    }
    let mut out1 = create_writer(&args.out1_path, args.gzip)?;
    let mut sorted_mod_types: Vec<String> = seen_mod_types.into_iter().collect();
    sorted_mod_types.sort();


    let mut out1_header = String::from("tx_name\ttx_pos\tn_read\tsum_probs\tn_nomod\twt_nomod");
    for m_type in &sorted_mod_types {
        out1_header.push_str(&format!("\tn_{}\twt_{}", m_type.to_lowercase(), m_type.to_lowercase()));
    }
    writeln!(out1, "{}", out1_header)?;

    for ((tx_id, pos), stats) in &modification_summary {
        let tx_name = &tx_names[*tx_id as usize];
        let mut line = format!("{}\t{}\t{}\t{:.4}\t{}\t{:.4}", tx_name, pos, stats.n_read, stats.sum_probs, stats.n_nomod, stats.wt_nomod);
        for m_type in &sorted_mod_types {
            let (n_m, wt_m) = stats.mods.get(m_type).cloned().unwrap_or((0, 0.0));
            line.push_str(&format!("\t{}\t{:.4}", n_m, wt_m));
        }
        writeln!(out1, "{}", line)?;
    }
    out1.flush()?;

    // --- Output Generation 2: Poly(A) Tail Distribution Summaries ---
    if args.verbose {
        eprintln!("[isolens] Writing Output 2 (PolyA Profiles)...");
    }
    let mut out2 = create_writer(&args.out2_path, args.gzip)?;
    writeln!(out2, "tx_name\tn_reads\tsum_weights\tpa_reads\tpa_weights\tpa_wlen\tprobs\tpa_lens")?;

    for (tx_id, stats) in &poly_a_summary {
        let tx_name = &tx_names[*tx_id as usize];
        let weighted_avg_len = if stats.pa_weights > 0.0 {
            stats.sum_weighted_pa_len / stats.pa_weights
        } else {
            0.0
        };

        let probs_str = stats.probs.iter().map(|p| format!("{:.4}", p)).collect::<Vec<_>>().join(",");
        let lens_str = stats.pa_lens.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(",");

        writeln!(
            out2,
            "{}\t{}\t{:.4}\t{}\t{:.4}\t{:.2}\t{}\t{}",
            tx_name, stats.n_reads, stats.sum_weights, stats.pa_reads, stats.pa_weights, weighted_avg_len, probs_str, lens_str
        )?;
    }
    out2.flush()?;

    Ok(())
}

fn parse_oarfish<P: AsRef<Path>>(path: P) -> Result<(Vec<String>, HashMap<String, Vec<TargetAssignment>, ahash::RandomState>, HashMap<String, u32, ahash::RandomState>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let decoder = Lz4Decoder::new(file);
    let reader = BufReader::new(decoder);
    let mut lines = reader.lines();

    let first_line = lines.next().ok_or("Empty oarfish file")??;
    let tokens: Vec<&str> = first_line.split_whitespace().collect();
    let num_transcripts: usize = tokens[0].parse()?;

    let mut tx_names = Vec::with_capacity(num_transcripts);
    let mut name_to_id = HashMap::with_hasher(ahash::RandomState::new());

    for id in 0..num_transcripts {
        let name = lines.next().ok_or("Malformed oarfish headers")??;
        let clean_name = name.trim().to_string();
        name_to_id.insert(clean_name.clone(), id as u32);
        tx_names.push(clean_name);
    }

    let mut prob_map = HashMap::with_hasher(ahash::RandomState::new());

    for line_result in lines {
        let line = line_result?;
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() { continue; }

        let read_id = tokens[0].to_string();
        let num_targets: usize = tokens[1].parse()?;

        let target_ids_tokens = &tokens[2..2 + num_targets];
        let prob_tokens = &tokens[2 + num_targets..2 + (2 * num_targets)];

        let mut assignments = Vec::with_capacity(num_targets);
        for i in 0..num_targets {
            let tx_idx: usize = target_ids_tokens[i].parse()?;
            let prob: f32 = prob_tokens[i].parse()?;

            assignments.push(TargetAssignment {
                tx_id: tx_idx as u32,
                prob,
            });
        }
        prob_map.insert(read_id, assignments);
    }

    Ok((tx_names, prob_map, name_to_id))
}

fn map_read_to_tx(cigar: &impl Cigar, tx_start_1: usize) -> Vec<Option<usize>> {
    let mut map = Vec::new();
    let mut curr_tx_1 = tx_start_1;

    for result in cigar.iter() {
        let op: Op = result.unwrap();
        let len = op.len();

        match op.kind() {
            Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                for _ in 0..len {
                    map.push(Some(curr_tx_1));
                    curr_tx_1 += 1;
                }
            }
            Kind::Insertion | Kind::SoftClip => {
                for _ in 0..len {
                    map.push(None);
                }
            }
            Kind::Deletion | Kind::Skip => {
                curr_tx_1 += len;
            }
            Kind::HardClip | Kind::Pad => {}
        }
    }
    map
}

struct Args {
    bam_path: String,
    oarfish_path: String,
    out1_path: String,
    out2_path: String,
    out_per_base: Option<String>,
    mod_threshold: f32,
    threads: usize,
    gzip: bool,
    verbose: bool,
}

fn parse_args() -> Result<Option<Args>, lexopt::Error> {
    use lexopt::prelude::*;

    let mut bam_path = None;
    let mut oarfish_path = None;
    let mut out1_path = None;
    let mut out2_path = None;
    let mut out_per_base = None;
    let mut mod_threshold = 0.95;
    let mut threads = 4;
    let mut gzip = false;
    let mut verbose = false;

    let mut parser = lexopt::Parser::from_env();

    while let Some(arg) = parser.next()? {
        match arg {
            Short('b') | Long("bam") => bam_path = Some(parser.value()?.string()?),
            Short('p') | Long("prob") => oarfish_path = Some(parser.value()?.string()?),
            Long("out1") => out1_path = Some(parser.value()?.string()?),
            Long("out2") => out2_path = Some(parser.value()?.string()?),
            Long("out-per-base") => out_per_base = Some(parser.value()?.string()?),
            Short('t') | Long("threshold") => mod_threshold = parser.value()?.parse::<f32>()?,
            Short('j') | Long("threads") => threads = parser.value()?.parse::<usize>()?,
            Short('z') | Long("gzip") => gzip = true,
            Short('v') | Long("verbose") => verbose = true,
            Short('h') | Long("help") => {
                print_help();
                return Ok(None);
            }
            _ => return Err(arg.unexpected()),
        }
    }

    let bam_path = bam_path.ok_or_else(|| lexopt::Error::Custom("Missing required -b/--bam".into()))?;
    let oarfish_path = oarfish_path.ok_or_else(|| lexopt::Error::Custom("Missing required -p/--prob".into()))?;
    let out1_path = out1_path.ok_or_else(|| lexopt::Error::Custom("Missing required --out1".into()))?;
    let out2_path = out2_path.ok_or_else(|| lexopt::Error::Custom("Missing required --out2".into()))?;

    Ok(Some(Args {
        bam_path,
        oarfish_path,
        out1_path,
        out2_path,
        out_per_base,
        mod_threshold,
        threads,
        gzip,
        verbose,
    }))
}
