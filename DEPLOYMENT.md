# GitHub Pages study setup

The fixed-video blind test can run entirely as a static GitHub Pages site.
GitHub Pages serves the participant UI and prepared summary clips; a small
Google Apps Script web app appends anonymous answers to a private Google
Sheet. No Hugging Face Space, Docker, Python server, or Gradio instance is
required for this version.

The static version is limited to the three prepared videos. It cannot process
new video uploads or summarize participant-submitted links. Summary letters
are randomized independently in each participant's browser; unlike the
server-backed version, the static site cannot counterbalance assignments
globally across participants.

## 1. Prepare authorized summary media

The original clips are embedded from the three public YouTube links. The
generated A/B summaries must be reachable by the Pages site. Before publishing
any video, confirm that you have permission to distribute the source-derived
summary clips.

If authorized, copy only the six summary files into this structure:

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

The files are currently under the matching directories in
`demo/study/stimuli/`. The static site references A/B media using these exact
relative paths. Do not commit or publish video files unless redistribution is
authorized. The current six summary clips total approximately 46 MB.

## 2. Create a private response spreadsheet

1. Create a Google Sheet and keep its sharing restricted to the researcher(s)
   who should view participant records. Do not publish the sheet or grant
   participants access.
2. From the Sheet, choose **Extensions → Apps Script**.
3. Replace the script editor contents with
   [`deployment/google-apps-script/Code.gs`](./deployment/google-apps-script/Code.gs).
   Set `SPREADSHEET_ID` to the ID in the Sheet URL between `/d/` and `/edit`.
4. Save the project, select **Deploy → New deployment**, and choose **Web app**.
   Set **Execute as** to your account and **Who has access** to **Anyone** so
   participants can submit without signing in. Google may require you to review
   and authorize the script.
5. Deploy and copy the Web app URL ending in `/exec`. The Apps Script code
   writes only validated study responses; it does not expose a read endpoint.
   The spreadsheet remains private.

The endpoint must be public for anonymous submissions, so it can receive
spam. The script validates a fixed set of videos and answer values, limits
comment length, ignores duplicate tester/video submissions, and escapes
spreadsheet formula prefixes in comments. This is not equivalent to
authenticated access control. Keep the sheet private and monitor submissions.

## 3. Configure and publish the study

1. Open `docs/index.html` and replace
   `SET_GOOGLE_APPS_SCRIPT_WEB_APP_URL` with the deployed `/exec` URL.
2. Copy the authorized six summary clips to `docs/media/` as shown above.
3. Review the consent and privacy text, especially the age and YouTube
   disclosures, to ensure it matches your institution's requirements.
4. Publish the intended changes to the repository's `main` branch. The
   workflow at `.github/workflows/pages.yml` publishes only `docs/`.
5. In GitHub, open **Settings → Pages** and choose **GitHub Actions** as the
   build and deployment source. Wait for the Pages workflow to pass.
6. Open the published Pages URL and submit a test response. Verify that the
   row appears in the private Sheet before sharing the participant link.

The usual project-site URL is
`https://<owner>.github.io/<repository>/`. The study records can be viewed
directly in the private Google Sheet. The Apps Script source is kept outside
`docs/`, so GitHub Pages does not publish it.

## Records format

The Sheet's `Responses` tab is initialized automatically and receives one row
per answered video with an anonymous browser-generated tester ID, timestamp,
video ID, randomized model-to-letter assignment, choice, ratings, tags,
optional comment, playback counts, source skip, and video order. It does not
collect a name or email. The researcher can download the sheet as CSV using
Google Sheets.
