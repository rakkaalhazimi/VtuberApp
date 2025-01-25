// import '@mediapipe/pose';
import '@tensorflow/tfjs-backend-webgl';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { PoseDetector } from '@tensorflow-models/pose-detection';


// BlazePose keypoint pairs to connect for drawing the skeleton
const blazePoseConnections = [
  // Head and Face
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  // Upper Body
  [11, 12], [11, 23], [23, 25], [25, 27], [27, 29], [29, 31],
  [12, 24], [24, 26], [26, 28], [28, 30], [30, 32],
  // Arms
  [11, 13], [13, 15], [12, 14], [14, 16],
  // Hands and Fingers
  [15, 17], [15, 19], [15, 21],
  [16, 18], [16, 20], [16, 22]
];

export enum BlazePosePoint {
  NOSE,
  LEFT_EYE_INNER,
  LEFT_EYE,
  LEFT_EYE_OUTER,
  RIGHT_EYE_INNER,
  RIGHT_EYE,
  RIGHT_EYE_OUTER,
  LEFT_EAR,
  RIGHT_EAR,
  MOUTH_LEFT,
  MOUTH_RIGHT,
  LEFT_SHOULDER,
  RIGHT_SHOULDER,
  LEFT_ELBOW,
  RIGHT_ELBOW,
  LEFT_WRIST,
  RIGHT_WRIST,
  LEFT_PINKY,
  RIGHT_PINKY,
  LEFT_INDEX,
  RIGHT_INDEX,
  LEFT_THUMB,
  RIGHT_THUMB,
  LEFT_HIP,
  RIGHT_HIP,
  LEFT_KNEE,
  RIGHT_KNEE,
  LEFT_ANKLE,
  RIGHT_ANKLE,
  LEFT_HEEL,
  RIGHT_HEEL,
  LEFT_FOOT_INDEX,
  RIGHT_FOOT_INDEX,
  // BODYCENTER,
  // FOREHEAD,
  // LEFTTHUMB,
  // LEFTHAND,
  // RIGHTTHUMB,
  // RIGHTHAND,
};


// MoveNet keypoint pairs to connect for drawing the skeleton
const movenetPoseConnections = [
  // Head and Face
  [0, 1], [1, 3], [0, 2], [2, 4],
  // Upper Body
  [5, 6], [5, 11], [11, 13], [13, 15], [6, 12], [12, 14], [14, 16],
  // Arms
  [5, 7], [7, 9], [6, 8], [8, 10]
];

export enum MovenetPosePoint {
  NOSE,
  LEFT_EYE,
  RIGHT_EYE,
  LEFT_EAR,
  RIGHT_EAR,
  LEFT_SHOULDER,
  RIGHT_SHOULDER,
  LEFT_ELBOW,
  RIGHT_ELBOW,
  LEFT_WRIST,
  RIGHT_WRIST,
  LEFT_HIP,
  RIGHT_HIP,
  LEFT_KNEE,
  RIGHT_KNEE,
  LEFT_ANKLE,
  RIGHT_ANKLE,
}

export type PosePoint = typeof BlazePosePoint | typeof MovenetPosePoint;

const POSE_CONFIDENCE_SCORE = 0.5;
const MODEL_INPUT_WIDTH = 640;
const MODEL_INPUT_HEIGHT = 480;

export class PoseEstimation {
  public videoWidth: number;
  public videoHeight: number;
  public detector: PoseDetector;
  public modelType: poseDetection.SupportedModels;
  
  constructor(videoHeight: number, videoWidth: number) {
    this.videoHeight = videoHeight;
    this.videoWidth = videoWidth;
  }
  
  async loadMovenetPoseEstimationDetector() {
    console.log('Loading movenet pose detector...');
    this.modelType = poseDetection.SupportedModels.MoveNet;
    let detectorConfig = {
      runtime: 'tfjs',
      enableSmoothing: true,
      modelType: 'SinglePose.Lightning',
    };
    this.detector = await poseDetection.createDetector(this.modelType, detectorConfig);
    console.log('Movenet pose detector is loaded');
    return this.detector;
  }
  
  async loadBlazePoseEstimationDetector() {
    console.log('Loading blaze pose detector...');
    this.modelType = poseDetection.SupportedModels.BlazePose;
    // Forget about tfjs, use mediapipe for better performance
    let detectorConfig = {
      runtime: 'mediapipe',
      enableSmoothing: true,
      solutionPath: './node_modules/@mediapipe/pose',
      modelType: 'lite',
    };
    this.detector = await poseDetection.createDetector(this.modelType, detectorConfig);
    console.log('Blaze pose detector is loaded');
    return this.detector;
  }
  
  async estimatePose(video: HTMLVideoElement) {
    let poses = await this.detector.estimatePoses(video, {flipHorizontal: false});
    
    // Normalize keypoints for movenet and blazepose model with mediapipe
    for (let currentPose of poses) {
      for (let keypoint of currentPose.keypoints) {
        // Assuming the model uses a default size like 640x480 for the input.
        keypoint.x = (keypoint.x / MODEL_INPUT_WIDTH) * this.videoWidth;
        keypoint.y = (keypoint.y / MODEL_INPUT_HEIGHT) * this.videoHeight;
      }
    }
    
    return poses;
  }
  
  async drawPoseLandmarks(ctx: CanvasRenderingContext2D, pose: poseDetection.Pose[]) {
    
    let currentPose = pose[0];
    
    if (!currentPose) {
      return;
    }
    
    // let logs: any[] = [];
    // let index = 0;
    // for (let keypoint of currentPose.keypoints) {
    //   logs.push(`${index} ${keypoint.name}`);
    //   index++;
    // }
    // console.log(logs.join('\n'));
    
    // Clear previous drawings
    ctx.clearRect(0, 0, this.videoWidth, this.videoHeight);
    
    // Draw keypoints
    for (let keypoint of currentPose.keypoints) {
      
      if (keypoint.score! < POSE_CONFIDENCE_SCORE) {
        continue;
      }
      
      // if (keypoint.name! != 'left_eye') {
      //   continue;
      // }
      
      // Draw circle for keypoint
      // console.log(keypoint.name, keypoint.score, keypoint.x, keypoint.y, keypoint.z!);
      let {x, y} = keypoint;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fillStyle = 'red';
      ctx.fill();
    }
    
    // Draw skeletons
    let poseConnections: number[][];
    
    switch (this.modelType) {
      
      case poseDetection.SupportedModels.BlazePose:
        poseConnections = blazePoseConnections;
        break;
        
      case poseDetection.SupportedModels.MoveNet:
        poseConnections = movenetPoseConnections;
        break;
        
      default:
        poseConnections = [];
        break;
    }
    
    for (let connections of poseConnections) {
      
      let [index1, index2] = connections;
      let keyPoint1 = currentPose.keypoints[index1];
      let keyPoint2 = currentPose.keypoints[index2];
      
      if (
        keyPoint1.score! < POSE_CONFIDENCE_SCORE || 
        keyPoint2.score! < POSE_CONFIDENCE_SCORE
      ) {
        continue;
      }
      
      ctx.beginPath();
      ctx.moveTo(keyPoint1.x, keyPoint1.y);
      ctx.lineTo(keyPoint2.x, keyPoint2.y);
      ctx.strokeStyle = 'green';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
  
}