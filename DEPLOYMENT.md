# GitHub Pages study setup

The prepared-video blind test is a static GitHub Pages site. It submits each
completed video's anonymous response as JSON to a single Google Forms
paragraph-response question. Responses appear as one row per video in the
Google Form's linked response spreadsheet. No Hugging Face Space, Docker,
Apps Script, Python server, or Gradio instance is required.

The static version is limited to the three prepared videos. It cannot process
new video uploads or summarize participant-submitted links. Summary letters
are randomized independently in each participant's browser, rather than
globally counterbalanced across participants.

## Google Form configuration

The current public form ID and paragraph question ID are configured in
`docs/index.html`. The form question must remain a **Paragraph** question and
its generated entry ID must remain `entry.1411265540`.

In Google Forms settings:

- Turn off email collection and **Limit to 1 response** so participants can
  respond without signing into Google.
- Keep any linked response spreadsheet private to the researcher/team.
- Do not change the form question or its entry ID without updating the study
  page to match.

One JSON response is submitted per completed video. Participants can stop
between videos; already-submitted answers remain in the form responses.
Google Forms' confirmation page loading in the embedded frame is used to
advance the study, but the browser cannot inspect Google's cross-origin
response body. Before recruitment, submit a full test session and confirm all
three rows appear in the response spreadsheet.

## Study media

The original clips are embedded from YouTube. The generated A/B summaries are
served by GitHub Pages in this structure:

```text
docs/
  media/
    video_1b7119f62c3c35f5/
      A.mp4
      B.mp4
    video_55a15a20b3b64e8d/
      A.mp4
      B.mp4
    video_ede296ea9c2cb625/
      A.mp4
      B.mp4
```

Before publishing the summary videos, confirm that their redistribution is
authorized. The six clips total approximately 46 MB. Avoid committing source
videos unless redistribution is authorized and necessary.

## Publish and test

The GitHub Pages workflow at `.github/workflows/pages.yml` publishes the
`docs/` directory. Pages must be configured to use **GitHub Actions** as the
publishing source in **Settings → Pages**. The project site URL is
`https://itsmilindsahu.github.io/IIITK_Wasserstien-Annotation/`.

After a study change is pushed, wait for the Pages deployment workflow to
succeed. Test a complete study session, verify three corresponding rows in
the Google Form's response spreadsheet, and only then share the study URL with
participants.

The response records contain an anonymous browser-generated tester ID,
timestamp, video ID, randomized model-to-letter assignment, choice, ratings,
tags, optional comment, playback counts, source skip, and video order. The page
does not ask for a name or email. Google Forms may still process standard
technical request metadata; ensure your participant information and consent
language accurately describe the data handling and comply with applicable
institutional requirements.
