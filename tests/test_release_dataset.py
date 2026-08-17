import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import release_dataset


class ReleaseDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_dir = self.root / "source"
        self.parts_dir = self.root / "parts"
        self.output_dir = self.root / "output"
        self.manifest_path = self.root / "release_manifest.json"
        self.source_dir.mkdir()

        self.sqlite_bytes = b"sqlite-content-0123456789"
        self.policy_bytes = b"policy-parquet"
        self.tasks_bytes = b"tasks-parquet"
        (self.source_dir / "repository_runtime.sqlite3").write_bytes(
            self.sqlite_bytes
        )
        (self.source_dir / "policy_evidence.parquet").write_bytes(
            self.policy_bytes
        )
        (self.source_dir / "tasks.parquet").write_bytes(self.tasks_bytes)

    def tearDown(self):
        self.temp_dir.cleanup()

    def build_manifest(self):
        return release_dataset.build_release_manifest(
            source_dir=self.source_dir,
            parts_dir=self.parts_dir,
            manifest_path=self.manifest_path,
            release_tag="evidence-agent-dataset-v1",
            repository="YuanCXC/swe-evidence",
            chunk_size=8,
        )

    def test_only_sqlite_is_split_and_parquet_files_remain_single_assets(self):
        manifest = self.build_manifest()
        files = {entry["path"]: entry for entry in manifest["files"]}

        sqlite_entry = files["repository_runtime.sqlite3"]
        self.assertEqual("parts", sqlite_entry["mode"])
        self.assertGreater(len(sqlite_entry["parts"]), 1)
        self.assertTrue(
            all(part["size"] <= 8 for part in sqlite_entry["parts"])
        )

        for parquet_name in ("policy_evidence.parquet", "tasks.parquet"):
            parquet_entry = files[parquet_name]
            self.assertEqual("single", parquet_entry["mode"])
            self.assertEqual(parquet_name, parquet_entry["asset"])
            self.assertNotIn("parts", parquet_entry)

        part_names = {path.name for path in self.parts_dir.iterdir()}
        self.assertTrue(part_names)
        self.assertTrue(
            all(name.startswith("repository_runtime.sqlite3.part-") for name in part_names)
        )

    def test_merge_restores_sqlite_byte_for_byte(self):
        self.build_manifest()

        outputs = release_dataset.merge_parts(
            manifest_path=self.manifest_path,
            parts_dir=self.parts_dir,
            output_dir=self.output_dir,
        )

        restored = self.output_dir / "repository_runtime.sqlite3"
        self.assertEqual([restored], outputs)
        self.assertEqual(self.sqlite_bytes, restored.read_bytes())

    def test_missing_part_is_rejected(self):
        manifest = self.build_manifest()
        first_part = manifest["files"][1]["parts"][0]["name"]
        (self.parts_dir / first_part).unlink()

        with self.assertRaises(release_dataset.IntegrityError):
            release_dataset.verify_parts(self.manifest_path, self.parts_dir)

    def test_corrupt_part_is_rejected_without_final_output(self):
        manifest = self.build_manifest()
        sqlite_entry = next(
            entry for entry in manifest["files"] if entry["mode"] == "parts"
        )
        first_part = self.parts_dir / sqlite_entry["parts"][0]["name"]
        first_part.write_bytes(b"corrupt!")

        with self.assertRaises(release_dataset.IntegrityError):
            release_dataset.merge_parts(
                self.manifest_path,
                self.parts_dir,
                self.output_dir,
            )

        final_path = self.output_dir / "repository_runtime.sqlite3"
        self.assertFalse(final_path.exists())
        self.assertFalse(final_path.with_name(final_path.name + ".partial").exists())

    def test_existing_output_is_not_overwritten(self):
        self.build_manifest()
        self.output_dir.mkdir()
        output = self.output_dir / "repository_runtime.sqlite3"
        output.write_bytes(b"keep-me")

        with self.assertRaises(FileExistsError):
            release_dataset.merge_parts(
                self.manifest_path,
                self.parts_dir,
                self.output_dir,
            )

        self.assertEqual(b"keep-me", output.read_bytes())

    def test_verify_files_checks_merged_and_single_assets(self):
        self.build_manifest()
        self.output_dir.mkdir()
        for source in self.source_dir.iterdir():
            (self.output_dir / source.name).write_bytes(source.read_bytes())

        verified = release_dataset.verify_files(
            self.manifest_path,
            self.output_dir,
        )

        self.assertEqual(3, len(verified))

    def test_manifest_is_stable_across_repeated_generation(self):
        self.build_manifest()
        first_bytes = self.manifest_path.read_bytes()
        first_manifest = json.loads(first_bytes)

        second_manifest = self.build_manifest()

        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_bytes, self.manifest_path.read_bytes())

    def test_upload_assets_include_parts_and_complete_parquet_files(self):
        self.build_manifest()

        assets = release_dataset.collect_upload_assets(
            self.manifest_path,
            self.source_dir,
            self.parts_dir,
        )
        asset_names = {asset.name for asset in assets}

        self.assertIn("policy_evidence.parquet", asset_names)
        self.assertIn("tasks.parquet", asset_names)
        self.assertNotIn("repository_runtime.sqlite3", asset_names)
        self.assertTrue(
            any(
                name.startswith("repository_runtime.sqlite3.part-")
                for name in asset_names
            )
        )

    def test_download_rejects_corrupt_existing_asset_before_calling_gh(self):
        self.build_manifest()
        download_dir = self.root / "download"
        download_dir.mkdir()
        (download_dir / "policy_evidence.parquet").write_bytes(b"corrupt")

        with self.assertRaises(release_dataset.IntegrityError):
            release_dataset.download_assets(
                self.manifest_path,
                download_dir,
                gh_bin="definitely-missing-gh",
            )

    def test_parallel_upload_skips_matching_remote_assets(self):
        manifest = self.build_manifest()
        policy_entry = next(
            entry
            for entry in manifest["files"]
            if entry["path"] == "policy_evidence.parquet"
        )
        upload_calls = []

        def fake_run_gh(_gh_bin, arguments):
            if arguments[:2] == ["release", "view"]:
                return SimpleNamespace(
                    stdout=json.dumps(
                        {
                            "assets": [
                                {
                                    "name": policy_entry["asset"],
                                    "size": policy_entry["size"],
                                }
                            ]
                        }
                    )
                )
            upload_calls.append(arguments)
            return SimpleNamespace(stdout="")

        with mock.patch.object(release_dataset, "_run_gh", side_effect=fake_run_gh):
            uploaded, skipped = release_dataset.upload_assets(
                self.manifest_path,
                self.source_dir,
                self.parts_dir,
                gh_bin="fake-gh",
                jobs=4,
            )

        self.assertEqual(["policy_evidence.parquet"], skipped)
        self.assertIn("tasks.parquet", uploaded)
        self.assertEqual(len(uploaded), len(upload_calls))


if __name__ == "__main__":
    unittest.main()
