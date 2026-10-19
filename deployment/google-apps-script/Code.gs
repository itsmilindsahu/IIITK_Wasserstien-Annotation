const SPREADSHEET_ID = "PASTE_YOUR_GOOGLE_SHEET_ID_HERE";
const SHEET_NAME = "Responses";
const RESPONSE_COLUMNS = [
  "tester_id",
  "video_id",
  "timestamp",
  "which_model_is_A",
  "choice",
  "chosen_model",
  "tags",
  "free_text",
  "likert_A",
  "likert_B",
  "played_A_count",
  "played_B_count",
  "first_watched",
  "seen_before",
  "age_range",
  "source_skipped",
  "video_order",
  "first_video_selection"
];
const VIDEO_IDS = [
  "video_1b7119f62c3c35f5",
  "video_55a15a20b3b64e8d",
  "video_ede296ea9c2cb625"
];
const ALLOWED_TAGS = [
  "Mood",
  "Story continuity",
  "Important moments",
  "Pacing",
  "Visual quality",
  "Other"
];

function doPost(event) {
  let submissionId = "";
  let lock;
  try {
    if (!event || !event.parameter || !event.parameter.payload) {
      throw new Error("Missing response payload.");
    }
    const response = JSON.parse(event.parameter.payload);
    submissionId = String(response.submission_id || "");
    validateResponse(response);

    lock = LockService.getScriptLock();
    lock.waitLock(10000);
    const sheet = getResponseSheet();
    const lastRow = sheet.getLastRow();
    if (lastRow > 1) {
      const existingTester = sheet
        .getRange(2, 1, lastRow - 1, 1)
        .createTextFinder(response.tester_id)
        .matchEntireCell(true)
        .findNext();
      if (
        existingTester &&
        sheet.getRange(existingTester.getRow(), 2).getValue() === response.video_id
      ) {
        return resultPage(submissionId, true);
      }
    }
    const choice = response.choice;
    const chosenModel = choice === "none"
      ? ""
      : choice === "A"
        ? response.which_model_is_A
        : response.which_model_is_A === "mean" ? "barycenter" : "mean";
    const row = [
      response.tester_id,
      response.video_id,
      new Date().toISOString(),
      response.which_model_is_A,
      choice,
      chosenModel,
      JSON.stringify(response.tags),
      safeCellText(response.free_text),
      response.likert_A,
      response.likert_B,
      response.played_A_count,
      response.played_B_count,
      response.first_watched,
      response.seen_before,
      response.age_range,
      response.source_skipped,
      response.video_order,
      response.first_video_selection
    ];
    sheet.appendRow(row);
    return resultPage(submissionId, true);
  } catch (error) {
    console.error(error);
    return resultPage(submissionId, false);
  } finally {
    if (lock && lock.hasLock()) {
      lock.releaseLock();
    }
  }
}

function getResponseSheet() {
  if (SPREADSHEET_ID === "PASTE_YOUR_GOOGLE_SHEET_ID_HERE") {
    throw new Error("Set SPREADSHEET_ID before deploying the web app.");
  }
  const spreadsheet = SpreadsheetApp.openById(SPREADSHEET_ID);
  let sheet = spreadsheet.getSheetByName(SHEET_NAME);
  if (!sheet) {
    sheet = spreadsheet.insertSheet(SHEET_NAME);
  }
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(RESPONSE_COLUMNS);
  }
  return sheet;
}

function validateResponse(response) {
  const isUuid = (value) =>
    typeof value === "string" &&
    /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
  const integerInRange = (value, minimum, maximum) =>
    Number.isInteger(value) && value >= minimum && value <= maximum;
  const ageRanges = [
    "",
    "18–24",
    "25–34",
    "35–44",
    "45–54",
    "55–64",
    "65 or older"
  ];
  const firstVideoChoices = [
    "Random order (recommended)",
    "Video 1",
    "Video 2",
    "Video 3"
  ];
  if (!isUuid(response.submission_id) || !isUuid(response.tester_id)) {
    throw new Error("Invalid anonymous response identifier.");
  }
  if (response.honeypot) {
    throw new Error("Automated submission rejected.");
  }
  if (!VIDEO_IDS.includes(response.video_id)) {
    throw new Error("Unknown study video.");
  }
  if (!["mean", "barycenter"].includes(response.which_model_is_A)) {
    throw new Error("Invalid randomized assignment.");
  }
  if (!["A", "B", "none"].includes(response.choice)) {
    throw new Error("Invalid response choice.");
  }
  const expectedModel = response.choice === "none"
    ? ""
    : response.choice === "A"
      ? response.which_model_is_A
      : response.which_model_is_A === "mean" ? "barycenter" : "mean";
  if (response.chosen_model !== expectedModel) {
    throw new Error("Response choice does not match its randomized assignment.");
  }
  if (
    !Array.isArray(response.tags) ||
    response.tags.some((tag) => !ALLOWED_TAGS.includes(tag)) ||
    new Set(response.tags).size !== response.tags.length
  ) {
    throw new Error("Invalid response tags.");
  }
  if (typeof response.free_text !== "string" || response.free_text.length > 1000) {
    throw new Error("Comment is too long.");
  }
  if (
    !integerInRange(response.likert_A, 1, 5) ||
    !integerInRange(response.likert_B, 1, 5) ||
    !integerInRange(response.played_A_count, 0, 1000) ||
    !integerInRange(response.played_B_count, 0, 1000)
  ) {
    throw new Error("Invalid ratings or playback counts.");
  }
  if (!["A", "B", "none"].includes(response.first_watched)) {
    throw new Error("Invalid playback order.");
  }
  if (!["", "Yes", "No"].includes(response.seen_before)) {
    throw new Error("Invalid familiarity response.");
  }
  if (!ageRanges.includes(response.age_range)) {
    throw new Error("Invalid age range.");
  }
  if (typeof response.source_skipped !== "boolean") {
    throw new Error("Invalid source-video skip value.");
  }
  if (!integerInRange(response.video_order, 1, 3)) {
    throw new Error("Invalid study-video order.");
  }
  if (!firstVideoChoices.includes(response.first_video_selection)) {
    throw new Error("Invalid first-video selection.");
  }
}

function safeCellText(value) {
  const trimmed = value.trim();
  return /^[=+\-@]/.test(trimmed) ? "'" + trimmed : trimmed;
}

function resultPage(submissionId, ok) {
  const message = JSON.stringify({
    type: "study-response-saved",
    submissionId: submissionId,
    ok: ok
  });
  return HtmlService.createHtmlOutput(
    "<!doctype html><script>parent.postMessage(" + message + ", '*');</script>"
  ).setXFrameOptionsMode(HtmlService.XFrameOptionsMode.ALLOWALL);
}
