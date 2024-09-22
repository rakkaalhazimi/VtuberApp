// import '@mediapipe/pose';
// import '@tensorflow/tfjs-backend-webgl';
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

export class PoseEstimation {
  public videoWidth: number;
  public videoHeight: number;
  public detector: PoseDetector;
  
  constructor(videoHeight: number, videoWidth: number) {
    this.videoHeight = videoHeight;
    this.videoWidth = videoWidth;
  }
  
  async loadPoseEstimationDetector() {
    let model = poseDetection.SupportedModels.BlazePose;
    let detectorConfig = {
      runtime: 'tfjs',
      enableSmoothing: true,
      modelType: 'full'
    };
    this.detector = await poseDetection.createDetector(model, detectorConfig);
    return this.detector;
  }
  
  async estimatePose(video: HTMLVideoElement) {
    let poses = await this.detector.estimatePoses(video, {flipHorizontal: false});
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
      
      if (keypoint.score! < 0.9) {
        continue;
      }
      
      // if (keypoint.name! != 'left_eye') {
      //   continue;
      // }
      
      // Draw circle for keypoint
      // console.log(keypoint.name, keypoint.score, keypoint.x, keypoint.y);
      let {x, y} = keypoint;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fillStyle = 'red';
      ctx.fill();
    }
    
    // Draw skeletons
    for (let connections of blazePoseConnections) {
      
      let [index1, index2] = connections;
      let keyPoint1 = currentPose.keypoints[index1];
      let keyPoint2 = currentPose.keypoints[index2];
      
      if (keyPoint1.score! < 0.9 || keyPoint2.score! < 0.9) {
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