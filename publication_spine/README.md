# Publication Spine

## Hash clarification for AoR `aor-20260209T040755Z`

Several PDFs in this publication_spine print:

- Bundle sha256: `c299b1b7a8ef77f25c3ebb326cb73f060b3c7176b6ea9eb402c97273dc3cf66c`

Clarification:
- The value above refers to the sealed GUM bundle identity (see `.../GUM_BUNDLE_v30_20260209T040755Z/bundle_sha256.txt`). It is not the sha256 of the master release zip.
- The canonical master release zip is:
  `gum/authority_archive/AOR_20260209T040755Z_0fc79a0/MARI_MASTER_RELEASE_20260209T040755Z_0fc79a0.zip`
  sha256:
  `c46392d1905c0b8a9305395dc7141af9c8cfc47c9b032d33e94d2eaca88c9c19`

Verification commands:

```bash
git checkout aor-20260209T040755ZCat gum/authority_archive/AOR_20260209T040755Z_0fc79a0/GUM_BUNDLE_v30_20260209T040755Z/bundle_sha256.txt
sha256sum gum/authority_archive/AOR_20260209T040755Z_0fc79a0/MARI_MASTER_RELEASE_20260209T040755Z_0fc79a0.zip
```

Note:
- PDFs are kept immutable to preserve their existing file hashes. This README is the authoritative clarification for hash interpretation.
