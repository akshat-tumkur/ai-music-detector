# Dataset acquisition

The target corpus is **440 tracks**: 220 human-composed tracks from MTG-Jamendo
and 220 AI-generated tracks from the pinned
[`ai-music4you/ai-generated-songs`](https://huggingface.co/datasets/ai-music4you/ai-generated-songs)
revision. The AI collection is much larger than this project needs, so the
acquisition script requests only a deterministic first 220-row sample.

MTG-Jamendo distributes music under Creative Commons licenses and requires
track-level attribution. The AI collection contains Suno-origin audio and does
not grant an automatic right to redistribute it; treat those files as local
research material unless the applicable terms explicitly permit sharing.
The current AI dataset revision is
`cf62db6dc87ca2cbab2ed5e1170bfa727a632013`.

## Build the manifest

```powershell
python -m src.acquire_dataset --ai 220
$env:JAMENDO_CLIENT_ID = "your-jamendo-client-id"
python -m src.acquire_dataset --human 220
```

The Jamendo API client ID is intentionally supplied through an environment
variable and is never written to the repository.

The current local corpus contains 220 AI-generated and 220 human tracks. Three
unavailable Jamendo URLs were replaced from a later API page so the manifest
contains exactly 440 downloadable files.

## Download locally

```powershell
python -m src.acquire_dataset --download
```

Raw audio is ignored by Git. Keep the manifest and downloaded files only on
machines where the source terms permit local research use. Preserve the
license and attribution fields before sharing any derived result.
