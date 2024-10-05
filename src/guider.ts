import { Face, Keypoint } from '@tensorflow-models/face-landmarks-detection';
import { Pose } from '@tensorflow-models/pose-detection';

import { FaceLandmark, LandmarkPoint } from './face-landmarks';
import { euclideanDistance2D } from './math';
import { Model } from './model';
import { MovenetPosePoint } from './pose-estimation';



const MIN_EAR_THRES = 0.2;
const MIN_MAR_THRES = 0.08;
const MAX_MAR_THRES = 0.3;

export class ModelMovementGuider {
  public model: Model;
  public faceLandmark: FaceLandmark;
  
  constructor(model: Model, faceLandmark: FaceLandmark) {
    this.model = model;
    this.faceLandmark = faceLandmark;
  }
  
  // Shift coordinate smoothly by incrementing by
  // small number
  shiftCoordinate(
    origin: number, 
    target: number,
    speed: number = 0.01,
    multiplier: number = 1,
  ) {
    
    let diff = Math.abs(origin - target * multiplier);
    if (diff <= speed) {
      return origin;
    }
    if (origin < target * multiplier) {
      return origin + speed;
    } else {
      return origin - speed;
    }
  }
  
  shiftCoordinateNew(
    origin: number, 
    target: number,
    speed: number = 0.03,
    diffThreshold: number = 0.02,
  ) {
    // Avoid random movement by limiting model movement
    // sensitivity.
    let diff = Math.abs(origin - target);
    if (diff <= diffThreshold) {
      return 0;
    }
    if (origin < target) {
      return Math.min(speed, diff);
    } else {
      return -Math.min(speed, diff);
    }
  }
  
  smoothMovement(
    newPosition: number, 
    previousPosition: number, 
    smoothingFactor: number = 0.2
  ) {
    return previousPosition + (newPosition - previousPosition) * smoothingFactor;
  }
  
  guideBlinking(faces: Face[]) {
    // Left eyelid
    let leftEyelidTopLeft = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, 
        LandmarkPoint.LEFT_EYELID_TOP_LEFT
    );
    let leftEyelidBottomLeft = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, 
        LandmarkPoint.LEFT_EYELID_BOTTOM_LEFT
    );
    let leftEyelidTopRight = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, 
        LandmarkPoint.LEFT_EYELID_TOP_RIGHT
    );
    let leftEyelidBottomRight = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, 
        LandmarkPoint.LEFT_EYELID_BOTTOM_RIGHT
    );
    
    // Right eyelid
    let rightEyelidTopLeft = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, 
        LandmarkPoint.RIGHT_EYELID_TOP_LEFT
    );
    let rightEyelidBottomLeft = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, 
        LandmarkPoint.RIGHT_EYELID_BOTTOM_LEFT
    );
    
    let rightEyelidTopRight = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, 
        LandmarkPoint.RIGHT_EYELID_TOP_RIGHT
    );
    let rightEyelidBottomRight = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, 
        LandmarkPoint.RIGHT_EYELID_BOTTOM_RIGHT
    );
    
    // Left Eye length
    let leftEyeStart = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, 
        LandmarkPoint.LEFT_EYELID_START
    );
    let leftEyeEnd = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, 
        LandmarkPoint.LEFT_EYELID_END
    );
    
    // Right Eye length
    let rightEyeStart = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, 
        LandmarkPoint.RIGHT_EYELID_START
    );
    let rightEyeEnd = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, 
        LandmarkPoint.RIGHT_EYELID_END
    );
    
    // Eye aspect ratio (EAR)
    let leftEAR = 
      this.faceLandmark.calculateAspectRatio(
        leftEyelidTopLeft.y,
        leftEyelidBottomLeft.y,
        leftEyelidTopRight.y,
        leftEyelidBottomRight.y,
        leftEyeStart.x,
        leftEyeEnd.x
    );
    
    let rightEAR = 
      this.faceLandmark.calculateAspectRatio(
        rightEyelidTopLeft.y,
        rightEyelidBottomLeft.y,
        rightEyelidTopRight.y,
        rightEyelidBottomRight.y,
        rightEyeStart.x,
        rightEyeEnd.x
    );
    
    // let ear = (leftEyeAspectRatio + rightEyeAspectRatio) / 2;
    // let ear = leftEAR;
    
    if (leftEAR < MIN_EAR_THRES || rightEAR < MIN_EAR_THRES) {
      this.model.morph('Blinking', 1);
    } else {
      this.model.morph('Blinking', 0);
    }
    
  }
  
  guideMouthMovement(faces: Face[]) {
    let mouthTopLeft = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, LandmarkPoint.MOUTH_TOP_LEFT  
    );
    let mouthBottomLeft = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, LandmarkPoint.MOUTH_BOTTOM_LEFT
    );
    
    let mouthTopRight = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, LandmarkPoint.MOUTH_TOP_RIGHT  
    );
    let mouthBottomRight = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, LandmarkPoint.MOUTH_BOTTOM_RIGHT
    );
    
    let mouthTopCenter = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, LandmarkPoint.MOUTH_TOP_CENTER
    );
    let mouthBottomCenter = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, LandmarkPoint.MOUTH_BOTTOM_CENTER
    );
    
    let mouthStart = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, LandmarkPoint.MOUTH_START
    );
    let mouthEnd = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, LandmarkPoint.MOUTH_END
    );
    
    // Mouth Aspect Ratio (MAR)
    let mar = this.faceLandmark.calculateAspectRatio(
      mouthTopLeft.y,
      mouthBottomLeft.y,
      mouthTopRight.y,
      mouthBottomRight.y,
      mouthStart.x,
      mouthEnd.x
    )
    
    // Mouth vertical gap
    let mouthVerticalGap = euclideanDistance2D(
      [mouthTopCenter.x, mouthTopCenter.y],
      [mouthBottomCenter.x, mouthBottomCenter.y]
    );
    
    // Mouth width
    let mouthWidth = euclideanDistance2D(
      [mouthStart.x, mouthStart.y],
      [mouthEnd.x, mouthEnd.y]
    );
    
    // Mouth curvature
    let mouthCurvature = mouthVerticalGap / mouthWidth;
    
    // Eye distance
    let leftEyeStart = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, 
        LandmarkPoint.LEFT_EYELID_START
    );
    
    let rightEyeStart = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, 
        LandmarkPoint.RIGHT_EYELID_START
    );
    
    let eyeDistance = euclideanDistance2D(
      [rightEyeStart.x, rightEyeStart.y],
      [leftEyeStart.x, leftEyeStart.y]
    );
    
    // Horizontal Aspect Ratio (HAR)
    // Eye distance is used for normalization
    let har = mouthWidth / eyeDistance;
    
    // Mouth movement
    // Vowel A
    // let mouthARatio = mar / MAX_MAR_THRES;
    let mouthARatio = Math.max(0, mar - MIN_MAR_THRES) / (MAX_MAR_THRES - MIN_MAR_THRES);
    if (mar > MIN_MAR_THRES) {
      this.model.morph('A', Math.min(mouthARatio, 1));
    } else {
      let currentValue = this.model.getMorphValue('A');
      this.model.morph('A', this.smoothMovement(0, currentValue, 0.1));
    }
    
    // Vowel I
    // When HAR is high but MAR is slightly low
    let mouthIRatio = Math.max(0, har - 1.3) / (1.4 - 1.3);
    if (har > 1.3 && mar > 0.04) {
      this.model.morph('I', Math.min(mouthIRatio, 1));
    } else {
      let currentValue = this.model.getMorphValue('I');
      this.model.morph('I', this.smoothMovement(0, currentValue, 0.1));
    }
    
    // Vowel U
    // When HAR is low but MAR is slightly high
    let mouthURatio = 0.5;
    if (har < 1.150 && mar > 0.05) {
      this.model.morph('U', mouthURatio);
    } else {
      let currentValue = this.model.getMorphValue('U');
      this.model.morph('U', this.smoothMovement(0, currentValue, 0.1));
    }
    
    // let x = document.getElementById('x-coordinate');
    // let y = document.getElementById('y-coordinate');
    // let z = document.getElementById('z-coordinate');
    // x!.innerHTML = mar.toFixed(3);
    // y!.innerHTML = har.toFixed(3);
    // z!.innerHTML = mouthCurvature.toFixed(3);
    // z!.innerHTML = leftEAR.toFixed(3);
    
    
  }
  
  guideHeadRotation(faces: Face[]) {
    let faceTop = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, LandmarkPoint.FACE_TOP
    );
    let faceBottom = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, LandmarkPoint.FACE_BOTTOM
    );
    let faceLeft = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, LandmarkPoint.FACE_LEFT
    );
    let faceRight = 
      this.faceLandmark.getLandmarkCoordinate(
        faces, LandmarkPoint.FACE_RIGHT
    );
    
    let horizontalDeltaX = faceLeft.x - faceRight.x;
    let horizontalDeltaY = faceLeft.y - faceRight.y;
    let horizontalDeltaZ = faceLeft.z! - faceRight.z!;
    let verticalDeltaY = faceTop.y - faceBottom.y;
    let verticalDeltaZ = faceTop.z! - faceBottom.z!;
    
    let roll = horizontalDeltaY / horizontalDeltaX;
    let yaw = horizontalDeltaZ / horizontalDeltaX;
    let pitch = verticalDeltaZ / verticalDeltaY;
    
    let head = this.model.boneDict['Head'];
    head.rotation.z = this.smoothMovement(roll, head.rotation.z);
    head.rotation.y = this.smoothMovement(-yaw, head.rotation.y);
    head.rotation.x = this.smoothMovement(pitch, head.rotation.x);
    
    // let euler = new THREE.Euler(pitch, -yaw, roll);
    // let quaternion = new THREE.Quaternion();
    // quaternion.setFromEuler(euler);
    // head.rotation.setFromQuaternion(quaternion);
    
    // let x = document.getElementById('x-coordinate');
    // let y = document.getElementById('y-coordinate');
    // let z = document.getElementById('z-coordinate');
    // x!.innerHTML = roll.toFixed(3);
    // y!.innerHTML = yaw.toFixed(3);
    // z!.innerHTML = pitch.toFixed(3);
    
  }
  
  guideUpperBodyMovement(faces: Face[], poses: Pose[]) {
    // let origin = 
    //   this.faceLandmark.getScaledLandmarkCoordinate(
    //     faces, LandmarkPoint.NOSE_MIDDLE
    // );
    
    let currentPose = poses[0];
    
    let leftShoulder = currentPose.keypoints[MovenetPosePoint.LEFT_SHOULDER];
    let rightShoulder = currentPose.keypoints[MovenetPosePoint.RIGHT_SHOULDER];
    let origin = {
      x: ((leftShoulder.x - rightShoulder.x) / 2) + rightShoulder.x,
      y: ((leftShoulder.y - rightShoulder.y) / 2) + rightShoulder.y,
      z: leftShoulder.z
    }
    
    // TODO: move the upper body based on shoulder
    // Scale the coordinate
    origin.x = (origin.x - 180) / 180;
    origin.y = (origin.y - 135) / 135;
    
    // coor.x = (coor.x - this.halfVideoWidth) / this.halfVideoWidth;
    // coor.y = (coor.y - this.halfVideoHeight) / this.halfVideoHeight;
    // coor.z = (coor.z! - this.initFaceZ) / this.initFaceZ;
    
    console.log(origin.x, origin.y);
    
    let upperBody = this.model.boneDict['Upper body'];
    let head = this.model.boneDict['Head'];
    
    // Rotate upper body left and right
    upperBody.rotation.z += this.shiftCoordinateNew(upperBody.rotation.z, origin.x);
    
    // Rotate upper body forward and backward;
    // upperBody.rotation.x += this.shiftCoordinateNew(upperBody.rotation.x, origin.z!);
    
    // Rotate head upward if we get too close
    // head.rotation.x += this.shiftCoordinateNew(head.rotation.x, -origin.z!);
    
    // Align head straight as the upper body rotate left and right
    // head.rotation.z += this.shiftCoordinateNew(head.rotation.z, -origin.x);
  }
  
  guideMovement(faces: Face[]) {
    if (!faces.length) {
      return;
    }
    this.guideBlinking(faces);
    // this.guideOpenMouth(faces);
    this.guideHeadRotation(faces);
    // this.guideUpperBodyMovement(faces);
  }
  
}