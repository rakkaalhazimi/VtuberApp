import '@tensorflow/tfjs-backend-webgl';
import { 
  createDetector,
  Face,
  FaceLandmarksDetector,
  SupportedModels,
  util 
} from '@tensorflow-models/face-landmarks-detection';

import { TRIANGULATION } from './triangulation';


export enum LandmarkPoint {
  NOSE_MIDDLE = 19,
  
  LEFT_EYELID_TOP = 386,
  LEFT_EYELID_BOTTOM = 374,
  LEFT_EYELID_TOP_LEFT = 387,
  LEFT_EYELID_BOTTOM_LEFT = 373,
  LEFT_EYELID_TOP_RIGHT = 385,
  LEFT_EYELID_BOTTOM_RIGHT = 380,
  LEFT_EYELID_START = 362,
  LEFT_EYELID_END = 263,
  
  RIGHT_EYELID_TOP = 159,
  RIGHT_EYELID_BOTTOM = 145,
  RIGHT_EYELID_TOP_LEFT = 160,
  RIGHT_EYELID_BOTTOM_LEFT = 144,
  RIGHT_EYELID_TOP_RIGHT = 158,
  RIGHT_EYELID_BOTTOM_RIGHT = 153,
  RIGHT_EYELID_START = 133,
  RIGHT_EYELID_END = 33,
  
  MOUTH_TOP_LEFT = 312,
  MOUTH_BOTTOM_LEFT = 317,
  MOUTH_TOP_RIGHT = 82,
  MOUTH_BOTTOM_RIGHT = 87,
  MOUTH_TOP_CENTER = 13,
  MOUTH_BOTTOM_CENTER = 14,
  MOUTH_START = 62,
  MOUTH_END = 292,
  
  FACE_TOP = 9,
  FACE_BOTTOM = 164,
  FACE_RIGHT = 123,
  FACE_LEFT = 352,
  
  P2 = 385,
  P3 = 387,
  P6 = 380,
  P5 = 373
}


export class FaceLandmark {
  public videoWidth: number;
  public videoHeight: number;
  public halfVideoWidth: number;
  public halfVideoHeight: number;
  public initFaceZ: number = -14;
  
  public detector: FaceLandmarksDetector;
  
  public readonly NUM_KEYPOINTS = 468;
  public readonly NUM_IRIS_KEYPOINTS = 5;
  public readonly LABEL_TO_COLOR = {
    lips: '#E0E0E0',
    leftEye: '#30FF30',
    leftEyebrow: '#30FF30',
    leftIris: '#30FF30',
    rightEye: '#FF3030',
    rightEyebrow: '#FF3030',
    rightIris: '#FF3030',
    faceOval: '#E0E0E0',
  };
  
  constructor(videoWidth: number, videoHeight: number) {
    this.videoWidth = videoWidth;
    this.videoHeight = videoHeight;
    this.halfVideoWidth = videoWidth / 2;
    this.halfVideoHeight = videoHeight / 2;
  }
  
  distance(a: number[], b: number[]) {
    return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
  }
  
  drawPath(
    ctx: CanvasRenderingContext2D, 
    points: number[][], 
    closePath: boolean,
  ) {
    const region = new Path2D();
    region.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
      const point = points[i];
      region.lineTo(point[0], point[1]);
    }
  
    if (closePath) {
      region.closePath();
    }
    ctx.stroke(region);
  }
  
  /**
   * Draw the keypoints on the video.
   * @param ctx 2D rendering context.
   * @param faces A list of faces to render.
   * @param triangulateMesh Whether or not to display the triangle mesh.
   * @param boundingBox Whether or not to display the bounding box.
   * 
   * ref: https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/demos/shared/util.js
   */
  drawFaceLandmarks(
    ctx: CanvasRenderingContext2D, 
    faces: Face[], 
    triangulateMesh: boolean = false, 
    boundingBox: boolean = false
  ) {
    faces.forEach((face) => {
      const keypoints =
          face.keypoints.map((keypoint) => [keypoint.x, keypoint.y]);
  
      if (boundingBox) {
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 1;
  
        const box = face.box;
        this.drawPath(
            ctx,
            [
              [box.xMin, box.yMin], [box.xMax, box.yMin], [box.xMax, box.yMax],
              [box.xMin, box.yMax]
            ],
            true);
      }
  
      if (triangulateMesh) {
        ctx.strokeStyle = '#00800';
        ctx.lineWidth = 0.5;
  
        for (let i = 0; i < TRIANGULATION.length / 3; i++) {
          const points = [
            TRIANGULATION[i * 3],
            TRIANGULATION[i * 3 + 1],
            TRIANGULATION[i * 3 + 2],
          ].map((index) => keypoints[index]);
  
          this.drawPath(ctx, points, true);
        }
      } else {
        ctx.fillStyle = '#00800';
  
        for (let i = 0; i < this.NUM_KEYPOINTS; i++) {
          const x = keypoints[i][0];
          const y = keypoints[i][1];
  
          ctx.beginPath();
          ctx.arc(x, y, 1 /* radius */, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
  
      if (keypoints.length > this.NUM_KEYPOINTS) {
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 1;
  
        const leftCenter = keypoints[this.NUM_KEYPOINTS];
        const leftDiameterY =
          this.distance(keypoints[this.NUM_KEYPOINTS + 4], keypoints[this.NUM_KEYPOINTS + 2]);
        const leftDiameterX =
          this.distance(keypoints[this.NUM_KEYPOINTS + 3], keypoints[this.NUM_KEYPOINTS + 1]);
  
        ctx.beginPath();
        ctx.ellipse(
          leftCenter[0], leftCenter[1], leftDiameterX / 2, leftDiameterY / 2, 0,
          0, 2 * Math.PI);
        ctx.stroke();
  
        if (keypoints.length > this.NUM_KEYPOINTS + this.NUM_IRIS_KEYPOINTS) {
          const rightCenter = keypoints[this.NUM_KEYPOINTS + this.NUM_IRIS_KEYPOINTS];
          const rightDiameterY = this.distance(
              keypoints[this.NUM_KEYPOINTS + this.NUM_IRIS_KEYPOINTS + 2],
              keypoints[this.NUM_KEYPOINTS + this.NUM_IRIS_KEYPOINTS + 4]);
          const rightDiameterX = this.distance(
              keypoints[this.NUM_KEYPOINTS + this.NUM_IRIS_KEYPOINTS + 3],
              keypoints[this.NUM_KEYPOINTS + this.NUM_IRIS_KEYPOINTS + 1]);
  
          ctx.beginPath();
          ctx.ellipse(
              rightCenter[0], rightCenter[1], rightDiameterX / 2,
              rightDiameterY / 2, 0, 0, 2 * Math.PI);
          ctx.stroke();
        }
      }
  
      const contours = util.getKeypointIndexByContour(SupportedModels.MediaPipeFaceMesh);
  
      for (const [label, contour] of Object.entries(contours)) {
        ctx.strokeStyle = this.LABEL_TO_COLOR[label];
        ctx.lineWidth = 1;
        const path = contour.map((index) => keypoints[index]);
        if (path.every(value => value != undefined)) {
          this.drawPath(ctx, path, false);
        }
      }
    });
  }
  
  async loadFaceLandmarksDetector() {
    console.log('Loading face landmarks detector...');
    //@ts-expect-error
    this.detector = await createDetector(SupportedModels.MediaPipeFaceMesh, {runtime: 'tfjs'});
    console.log('Face landmarks detector loaded.');
    return this.detector;
  }
  
  async estimateFaces(video: HTMLVideoElement) {
    let faces: Face[] = 
      await this.detector.estimateFaces(video, {flipHorizontal: false});
    return faces;
  }
  
  getLandmarkCoordinate(
    faces: Face[], 
    point: LandmarkPoint
  ) {
    let currentFace = faces[0];
    let coor = currentFace.keypoints[point];
    return coor;
  }
  
  getScaledLandmarkCoordinate(
    faces: Face[], 
    point: LandmarkPoint
  ) {
    
    
    let currentFace = faces[0];
    let coor = currentFace.keypoints[point];
    
    if (!this.initFaceZ) {
      this.initFaceZ = coor.z!;
    }
    
    coor.x = (coor.x - this.halfVideoWidth) / this.halfVideoWidth;
    coor.y = (coor.y - this.halfVideoHeight) / this.halfVideoHeight;
    coor.z = (coor.z! - this.initFaceZ) / this.initFaceZ;
    
    // let pointX = document.getElementById('x-coordinate');
    // let pointY = document.getElementById('y-coordinate');
    // let pointZ = document.getElementById('z-coordinate');
    
    // pointX!.innerHTML = coor.x.toFixed(3);
    // pointY!.innerHTML = coor.y.toFixed(3);
    // pointZ!.innerHTML = coor.z!.toFixed(3);
    
    return coor;
  }
  
  // Calculate aspect ratio by dividing width and height
  //
  // ref: https://medium.com/analytics-vidhya/eye-aspect-ratio-ear-and-drowsiness-detector-using-dlib-a0b2c292d706
  calculateAspectRatio(
    topLeft: number, 
    bottomLeft: number, 
    topRight: number, 
    bottomRight: number, 
    left: number, 
    right: number
  ) {
    let deltaLeft = Math.abs(topLeft - bottomLeft);
    let deltaRight = Math.abs(topRight - bottomRight);
    let length = Math.abs(left - right);
    
    let aspectRatio = (deltaLeft + deltaRight) / (2 * length);
    return aspectRatio;
  }
  
}
