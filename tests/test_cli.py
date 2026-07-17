"""Tests for the isolens CLI entry point and subcommand dispatch."""

from click.testing import CliRunner

from isolens import __version__

# Import the main CLI group — lazy imports inside subcommands are not
# triggered until invoked, so importing the group is safe / cheap.
from isolens.cli import main  # noqa: E402


class TestCliBasic:
    """Basic smoke tests for the CLI group."""

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert all(
            name in result.output
            for name in [
                "mod_scan",
                "mod_sites",
                "mod_corr",
                "mod_gene",
                "mod_dmc",
                "mod_dmt",
                "mod_dmcg",
                "polya_calc",
                "polya_merge",
                "polya_gene",
                "polya_bimodal",
                "polya_dpc",
                "polya_dpt",
            ]
        )

    def test_help_short_flag(self) -> None:
        """``-h`` is an alias for ``--help`` on the root group."""
        runner = CliRunner()
        result = runner.invoke(main, ["-h"])
        assert result.exit_code == 0
        assert "mod_scan" in result.output

    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_no_subcommand_shows_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, [])
        # click exits with 2 when no subcommand is given, but still
        # prints usage/help information.
        assert result.exit_code == 2
        assert "Usage:" in result.output or "Commands:" in result.output

    def test_invalid_subcommand(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["not-a-command"])
        assert result.exit_code == 2


class TestSubcommandHelp:
    """``--help`` and ``-h`` work on representative subcommands."""

    def test_mod_scan_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["mod_scan", "--help"])
        assert result.exit_code == 0
        assert "--bam" in result.output
        assert "--oarfish" in result.output

    def test_mod_scan_help_short_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["mod_scan", "-h"])
        assert result.exit_code == 0
        assert "--mod-cutoff" in result.output

    def test_mod_sites_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["mod_sites", "--help"])
        assert result.exit_code == 0
        assert "--h5" in result.output
        assert "--gtf" in result.output

    def test_mod_dmc_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["mod_dmc", "--help"])
        assert result.exit_code == 0
        assert "--h5-1" in result.output
        assert "--h5-2" in result.output
        assert "--sites-1" in result.output
        assert "--sites-2" in result.output

    def test_polya_calc_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["polya_calc", "--help"])
        assert result.exit_code == 0
        assert "--oarfish" in result.output
        assert "--bam" in result.output

    def test_polya_dpc_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["polya_dpc", "--help"])
        assert result.exit_code == 0
        assert "--condition1" in result.output
        assert "--condition2" in result.output

    def test_polya_dpt_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["polya_dpt", "--help"])
        assert result.exit_code == 0
        assert "--gtf" in result.output
        assert "--min-pareads" in result.output

    def test_polya_bimodal_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["polya_bimodal", "--help"])
        assert result.exit_code == 0
        assert "--min-length" in result.output
        assert "--min-ess" in result.output
        assert "--kde-prominence" in result.output


class TestSubcommandValidation:
    """Missing required options produce exit code 2."""

    def test_mod_scan_missing_bam(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["mod_scan", "-o", "out.h5"])
        assert result.exit_code == 2

    def test_mod_dmc_missing_required(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["mod_dmc"])
        assert result.exit_code == 2

    def test_polya_calc_missing_required(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["polya_calc"])
        assert result.exit_code == 2
