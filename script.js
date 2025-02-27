let detector;
let detectorConfig;
let poses;
let video;
let model;
let skeleton = true;
let reps = 0;
let stage = "Initial";
let lastCounterUpdate = Date.now();
const UP_TOLERANCE = 5;
const DOWN_TOLERANCE = 5;
const MIN_TIME_BETWEEN_COUNTS = 250; // 1 second

async function init() {
  detectorConfig = { modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER };
  detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
  edges = {
    "5,7": "m", "7,9": "m", "6,8": "c", "8,10": "c",
    "5,6": "y", "5,11": "m", "6,12": "c", "11,12": "y",
    "11,13": "m", "13,15": "m", "12,14": "c", "14,16": "c"
  };
  await getPoses();
}

async function videoReady() {
  console.log("Video ready");
}

async function setup() {
  var msg = new SpeechSynthesisUtterance("Loading, please wait...");
  window.speechSynthesis.speak(msg);

  createCanvas(640, 480);
  video = createCapture(VIDEO, videoReady);
  video.hide();

  await init();
}

async function getPoses() {
  poses = await detector.estimatePoses(video.elt);
  setTimeout(getPoses, 0);
}

function draw() {
  background(220);
  translate(width, 0);
  scale(-1, 1);
  image(video, 0, 0, video.width, video.height);

  drawKeypoints();
  if (skeleton) drawSkeleton();

  // Display the push-up counter
  fill(255);
  strokeWeight(2);
  stroke(51);
  translate(width, 0);
  scale(-1, 1);
  textSize(40);

  if (poses && poses.length > 0) {
    let pushupString = Push-ups: ${reps};
    text(pushupString, 100, 90);
  } else {
    text("Loading, please wait...", 100, 90);
  }
}

function drawKeypoints() {
  if (poses && poses.length > 0) {
    for (let kp of poses[0].keypoints) {
      const { x, y, score } = kp;
      if (score > 0.3) {
        fill(255);
        stroke(0);
        strokeWeight(4);
        circle(x, y, 16);
      }
    }
    trackPushUp();
  }
}

function drawSkeleton() {
  let confidence_threshold = 0.5;

  if (poses && poses.length > 0) {
    for (const [key, value] of Object.entries(edges)) {
      const p = key.split(",");
      const p1 = parseInt(p[0]);
      const p2 = parseInt(p[1]);

      const keypoint1 = poses[0].keypoints[p1];
      const keypoint2 = poses[0].keypoints[p2];

      if (keypoint1.score > confidence_threshold && keypoint2.score > confidence_threshold) {
        strokeWeight(2);
        stroke("rgb(0, 255, 0)");
        line(keypoint1.x, keypoint1.y, keypoint2.x, keypoint2.y);
      }
    }
  }
}

// *New Function for Push-Up Tracking*
function trackPushUp() {
  if (!poses || poses.length === 0) return;

  let leftShoulder = poses[0].keypoints[5];
  let leftElbow = poses[0].keypoints[7];
  let leftWrist = poses[0].keypoints[9];

  if (leftShoulder.score > 0.3 && leftElbow.score > 0.3 && leftWrist.score > 0.3) {
    let angle = calculateAngle(leftShoulder, leftElbow, leftWrist);

    let upThreshold = 150 - UP_TOLERANCE;
    let downThreshold = 70 + DOWN_TOLERANCE;

    if (angle > upThreshold) {
      stage = "Up";
    } else if (angle < downThreshold && stage === "Up") {
      stage = "Down";
      let currentTime = Date.now();
      if (currentTime - lastCounterUpdate > MIN_TIME_BETWEEN_COUNTS) {
        reps++;
        lastCounterUpdate = currentTime;
        speakCount(reps);
      }
    }
  }
}

// *Helper Function: Calculate Angle Between Three Points*
function calculateAngle(a, b, c) {
  let A = [a.x, a.y];
  let B = [b.x, b.y];
  let C = [c.x, c.y];

  let radians = Math.atan2(C[1] - B[1], C[0] - B[0]) - Math.atan2(A[1] - B[1], A[0] - B[0]);
  let angle = Math.abs((radians * 180.0) / Math.PI);
  return angle > 180 ? 360 - angle : angle;
}

// *Helper Function: Speak Count*
function speakCount(count) {
  let msg = new SpeechSynthesisUtterance(${count});
  window.speechSynthesis.speak(msg);
}