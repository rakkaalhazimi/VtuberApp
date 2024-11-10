import * as THREE from 'three';
import { Face, Keypoint } from '@tensorflow-models/face-landmarks-detection';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { Pose } from '@tensorflow-models/pose-detection';

import { FaceLandmark, LandmarkPoint } from './face-landmarks';
import { 
  euclideanDistance2D, 
  euclideanDistance3D, 
  gradient2D, 
  angleOfTriangle2D, 
  getSkewSymmetricMatrix, 
  getRotationMatrix, 
  crossProduct,
  rotationMatrixToEulerAnglesNew } from './math';
import { showXValue, showYValue, showZValue } from './metric';
import { Model } from './model';
import { BlazePosePoint, MovenetPosePoint, PosePoint, PoseEstimation } from './pose-estimation';



const MIN_EAR_THRES = 0.2;

const MIN_MAR_VOWEL_A = 0.08;
const MAX_MAR_VOWEL_A = 0.2;
const MIN_HAR_VOWEL_A = 0;

const MIN_MAR_VOWEL_I = 0.04;
const MAX_MAR_VOWEL_I = 0.12;
const MIN_HAR_VOWEL_I = 1.3;
const MAX_HAR_VOWEL_I = 1.4;

const MIN_MAR_VOWEL_U = 0.08;
const MAX_MAR_VOWEL_U = 0.12;
const MAX_HAR_VOWEL_U = 1.150;





export class ModelMovementGuider {
  public model: Model;
  public faceLandmark: FaceLandmark;
  public poseEstimation: PoseEstimation;
  public posePoint: PosePoint;
  
  constructor(
    model: Model, 
    faceLandmark: FaceLandmark, 
    poseEstimation: PoseEstimation
  ) {
    this.model = model;
    this.faceLandmark = faceLandmark;
    this.poseEstimation = poseEstimation;
    
    if (!this.poseEstimation.modelType) {
      throw new Error(`poseEstimation ModelType is undefined, please load pose estimation detector first`);
    }
    
    switch (this.poseEstimation.modelType) {
      case (poseDetection.SupportedModels.BlazePose):
        this.posePoint = BlazePosePoint;
        break;
      
      case (poseDetection.SupportedModels.MoveNet):
        this.posePoint = MovenetPosePoint
        break;
    }
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
    let mouthARatio = 
      Math.max(0, mar - MIN_MAR_VOWEL_A) / 
      (MAX_MAR_VOWEL_A - MIN_MAR_VOWEL_A);
      
    if (mar > MIN_MAR_VOWEL_A) {
      this.model.morph('A', Math.min(mouthARatio, 1));
    } else {
      let currentValue = this.model.getMorphValue('A');
      this.model.morph('A', this.smoothMovement(0, currentValue, 0.1));
    }
    
    // Vowel I
    // When HAR is high but MAR is slightly low
    let mouthIRatio = 
      Math.max(0, har - MIN_HAR_VOWEL_I) / 
      (MAX_HAR_VOWEL_I - MIN_HAR_VOWEL_I);
      
    if (
      har > MIN_HAR_VOWEL_I && 
      mar > MIN_MAR_VOWEL_I && 
      mar < MAX_MAR_VOWEL_I
    ) {
      this.model.morph('I', Math.min(mouthIRatio, 1));
    } else {
      let currentValue = this.model.getMorphValue('I');
      this.model.morph('I', this.smoothMovement(0, currentValue, 0.1));
    }
    
    // Vowel U
    // When HAR is low but MAR is slightly high
    let mouthURatio = 0.5;
    if (har < MAX_HAR_VOWEL_U && mar > MIN_MAR_VOWEL_U) {
      this.model.morph('U', mouthURatio);
    } else {
      let currentValue = this.model.getMorphValue('U');
      this.model.morph('U', this.smoothMovement(0, currentValue, 0.1));
    }
    
    // Smiling
    let smileRatio = Math.max(0, har - 1.21) / (1.4 - 1.21);
    if (mar < 0.025) {
      this.model.morph('Grin', smileRatio);
    } else {
      let currentValue = this.model.getMorphValue('Grin');
      this.model.morph('Grin', this.smoothMovement(0, currentValue, 0.1));
    }
    
    
    
    
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
  
  guideShoulderMovement(poses: Pose[]) {
    let currentPose = poses[0];
    
    let leftShoulderPose = currentPose.keypoints[this.posePoint.LEFT_SHOULDER];
    let leftShoulder3DPose = currentPose.keypoints3D![this.posePoint.LEFT_SHOULDER];
    let rightShoulderPose = currentPose.keypoints[this.posePoint.RIGHT_SHOULDER];
    let rightShoulder3DPose = currentPose.keypoints3D![this.posePoint.RIGHT_SHOULDER];
    
    let leftElbowPose = currentPose.keypoints[this.posePoint.LEFT_ELBOW];
    let leftElbow3DPose = currentPose.keypoints3D![this.posePoint.LEFT_ELBOW];
    let rightElbowPose = currentPose.keypoints[this.posePoint.RIGHT_ELBOW];
    let rightElbow3DPose = currentPose.keypoints3D![this.posePoint.RIGHT_ELBOW];
    
    let leftWristPose = currentPose.keypoints[this.posePoint.LEFT_WRIST];
    let rightWristPose = currentPose.keypoints[this.posePoint.RIGHT_WRIST];
    
    let leftShoulder = this.model.boneDict['Left shoulder'];
    let rightShoulder = this.model.boneDict['Right shoulder'];
    
    let leftArm = this.model.boneDict['Left arm'];
    let rightArm = this.model.boneDict['Right arm'];
    
    let leftElbow = this.model.boneDict['Left elbow'];
    let rightElbow = this.model.boneDict['Right elbow'];
    
    // [Shoulder Up and Down]
    // Raise shoulder up and down
    let shoulderGradient = gradient2D(
      [leftShoulderPose.x, leftShoulderPose.y],
      [rightShoulderPose.x, rightShoulderPose.y]
    );
    
    // leftShoulder.rotation.z = this.smoothMovement(shoulderGradient, leftShoulder.rotation.z, 0.4);
    // rightShoulder.rotation.z = this.smoothMovement(shoulderGradient, rightShoulder.rotation.z, 0.4);
    // [End Shoulder Up and Down]
    
    // [Arms Up and Down]
    // Rotate arms up and down based on shoulder angle
    let leftShoulderAngleZ = angleOfTriangle2D(
      [rightShoulderPose.x, rightShoulderPose.y],
      [leftShoulderPose.x, leftShoulderPose.y],
      [leftElbowPose.x, leftElbowPose.y],
      true  // include obscute angle
    );
    
    let rightShoulderAngleZ = angleOfTriangle2D(
      [leftShoulderPose.x, leftShoulderPose.y],
      [rightShoulderPose.x, rightShoulderPose.y],
      [rightElbowPose.x, rightElbowPose.y],
      true  // include obscute angle
    );
    
    // Adjust the angle with 90 degree from human normal pose
    // and 45 degree from model default hand pose.
    // leftArm.rotation.z = 
    //   this.smoothMovement(
    //     -leftShoulderAngleZ - (Math.PI / 2) - (Math.PI / 4),
    //     leftArm.rotation.z,
    //     0.1
    // );
    // rightArm.rotation.z = 
    //   this.smoothMovement(
    //     -rightShoulderAngleZ + (Math.PI / 2) + (Math.PI / 4),
    //     rightArm.rotation.z,
    //     0.1
    // );
    // [End Arms Up and Down]
    
    // [Arms Forward and Backward]
    // Bend arms forward and backward based on elbow angle.
    // TODO: Take a note that the rotation X and Y will follow the direction of the vector.
    //       If you expect rotation X to go forward and backward when hands is down.
    //       You might find that when the hands is a middle up, rotating X will only twist the arm.
    
    let shoulderDistance = euclideanDistance2D(
      [leftShoulder3DPose.x, leftShoulder3DPose.y],
      [rightShoulder3DPose.x, rightShoulder3DPose.y]
    );
    
    // Create point below shoulder
    let leftBelowShoulder3DPose = {
      x: 0.1,
      y: (shoulderDistance / 2),
      z: 0
    };
    
    let leftBelowShoulderLength = euclideanDistance3D(
      [0, 0, 0],
      [leftBelowShoulder3DPose.x, leftBelowShoulder3DPose.y, leftBelowShoulder3DPose.z]
    )
    
    leftBelowShoulder3DPose.x /= leftBelowShoulderLength;
    leftBelowShoulder3DPose.y /= leftBelowShoulderLength;
    leftBelowShoulder3DPose.z /= leftBelowShoulderLength;
    
    let leftElbow3DPoseRelative = {
      x: leftElbow3DPose.x - leftShoulder3DPose.x,
      y: leftElbow3DPose.y - leftShoulder3DPose.y,
      z: leftElbow3DPose.z! - leftShoulder3DPose.z!,
    }
    
    let leftElbowLength = euclideanDistance3D(
      [0, 0, 0],
      [leftElbow3DPoseRelative.x, leftElbow3DPoseRelative.y, leftElbow3DPoseRelative.z]
    )
    
    leftElbow3DPoseRelative.x /= leftElbowLength;
    leftElbow3DPoseRelative.y /= leftElbowLength;
    leftElbow3DPoseRelative.z /= leftElbowLength;
    
    let axisOfRotation = {
      x: (leftBelowShoulder3DPose.y * leftElbow3DPoseRelative.z) - (leftBelowShoulder3DPose.z * leftElbow3DPoseRelative.y),
      y: (leftBelowShoulder3DPose.z * leftElbow3DPoseRelative.x) - (leftBelowShoulder3DPose.x * leftElbow3DPoseRelative.z),
      z: (leftBelowShoulder3DPose.x * leftElbow3DPoseRelative.y) - (leftBelowShoulder3DPose.y * leftElbow3DPoseRelative.x),
    };
    
    let rotationAngle = 
      (leftBelowShoulder3DPose.x * leftElbow3DPoseRelative.x) +
      (leftBelowShoulder3DPose.y * leftElbow3DPoseRelative.y) +
      (leftBelowShoulder3DPose.z * leftElbow3DPoseRelative.z);
      
    // let quaternion = new THREE.Quaternion();
    // quaternion.setFromAxisAngle(new THREE.Vector3(
    //   axisOfRotation.x, axisOfRotation.y, axisOfRotation.z),
    //   rotationAngle
    // );
    // leftArm.setRotationFromQuaternion(quaternion);
    
    leftArm.rotation.x = this.smoothMovement(axisOfRotation.x, leftArm.rotation.x, 0.1);
    // leftArm.rotation.y = this.smoothMovement(axisOfRotation.y, leftArm.rotation.y, 0.1);
    leftArm.rotation.z = this.smoothMovement(-axisOfRotation.z, leftArm.rotation.z, 0.1);
    
    let leftShoulderAngleX = angleOfTriangle2D(
      [leftBelowShoulder3DPose.z!, leftBelowShoulder3DPose.y],
      [leftShoulder3DPose.z!, leftShoulder3DPose.y],
      [leftElbow3DPose.z!, leftElbow3DPose.y],
      // true  // include obscute angle
    );
    
    let rightShoulderAngleX = angleOfTriangle2D(
      [rightElbow3DPose.y, rightElbow3DPose.z!],
      [rightShoulder3DPose.y, rightShoulder3DPose.z!],
      [rightElbow3DPose.y, rightShoulder3DPose.z!],
      // true  // include obscute angle
    );
    
    // leftArm.rotation.x = this.smoothMovement(-leftShoulderAngleX, leftArm.rotation.x, 0.1);
    // rightArm.rotation.x = this.smoothMovement(-rightShoulderAngleX, rightArm.rotation.x, 0.1);
    // [End Arms Forward and Backward]
    
    // [Arms Inward and Outward]
    // Bend arms inward and outward based on elbow angle.
    let leftShoulderAngleY = angleOfTriangle2D(
      [rightShoulder3DPose.z!, rightShoulder3DPose.x],
      [leftShoulder3DPose.z!, leftShoulder3DPose.x],
      [leftElbow3DPose.z!, leftShoulder3DPose.x],
      // true  // include obscute angle
    );
    
    let rightShoulderAngleY = angleOfTriangle2D(
      [rightElbow3DPose.z!, rightElbow3DPose.x],
      [rightShoulder3DPose.z!, rightShoulder3DPose.x],
      [rightElbow3DPose.z!, rightShoulder3DPose.x],
      // true  // include obscute angle
    );
    
    // leftArm.rotation.y = this.smoothMovement(leftShoulderAngleY, leftArm.rotation.y, 0.1);
    // rightArm.rotation.y = this.smoothMovement(rightShoulderAngleY, rightArm.rotation.y, 0.1);
    
    // [Elbow Rotation]
    // Rotate elbow
    let leftElbowAngleX = angleOfTriangle2D(
      [leftShoulderPose.y, leftShoulderPose.z!],
      [leftElbowPose.y, leftElbowPose.z!],
      [leftWristPose.y, leftWristPose.z!],
      true  // include obscute angle
    );
    
    let rightElbowAngleX = angleOfTriangle2D(
      [rightShoulderPose.y, rightShoulderPose.z!],
      [rightElbowPose.y, rightElbowPose.z!],
      [rightWristPose.y, rightWristPose.z!],
      true  // include obscute angle
    );
    
    let leftElbowAngle = angleOfTriangle2D(
      [leftShoulderPose.x, leftShoulderPose.y],
      [leftElbowPose.x, leftElbowPose.y],
      [leftWristPose.x, leftWristPose.y],
      true
    );
    
    let rightElbowAngle = angleOfTriangle2D(
      [rightShoulderPose.x, rightShoulderPose.y],
      [rightElbowPose.x, rightElbowPose.y],
      [rightWristPose.x, rightWristPose.y],
      true
    );
    
    // When angle is 180 degree, value is 0
    // 
    
    // leftElbow.rotation.x = 
    //   this.smoothMovement(
    //     leftElbowAngle - Math.PI, 
    //     leftElbow.rotation.x, 
    //     0.1
    // );
    // leftElbow.rotation.y = 
    //   this.smoothMovement(
    //     leftElbowAngle - Math.PI, 
    //     leftElbow.rotation.y, 
    //     0.1
    // );
    
    // rightElbow.rotation.x = rightElbowAngle - Math.PI;
    // rightElbow.rotation.y = -(rightElbowAngle - Math.PI);
    
    // Lower the arms as the shoulder go up
    
    // if (shoulderGradient > 0) {
    //   leftArm.rotation.z = this.smoothMovement(-0.8, leftArm.rotation.z, 0.1);
    //   rightArm.rotation.z = this.smoothMovement(0.5, rightArm.rotation.z, 0.1);
    // } else {
    //   leftArm.rotation.z = this.smoothMovement(-0.5, leftArm.rotation.z, 0.1);
    //   rightArm.rotation.z = this.smoothMovement(0.8, rightArm.rotation.z, 0.1);
    // }
    
    showXValue(axisOfRotation.x);
    showYValue(axisOfRotation.y);
    showZValue(axisOfRotation.z);
  }
  
  guideShoulderMovementNew(poses: Pose[]) {
    let currentPose = poses[0];
    
    let leftShoulderPose = currentPose.keypoints[this.posePoint.LEFT_SHOULDER];
    let leftShoulder3DPose = currentPose.keypoints3D![this.posePoint.LEFT_SHOULDER];
    let rightShoulderPose = currentPose.keypoints[this.posePoint.RIGHT_SHOULDER];
    let rightShoulder3DPose = currentPose.keypoints3D![this.posePoint.RIGHT_SHOULDER];
    
    let leftElbowPose = currentPose.keypoints[this.posePoint.LEFT_ELBOW];
    let leftElbow3DPose = currentPose.keypoints3D![this.posePoint.LEFT_ELBOW];
    let rightElbowPose = currentPose.keypoints[this.posePoint.RIGHT_ELBOW];
    let rightElbow3DPose = currentPose.keypoints3D![this.posePoint.RIGHT_ELBOW];
    
    let leftWristPose = currentPose.keypoints[this.posePoint.LEFT_WRIST];
    let rightWristPose = currentPose.keypoints[this.posePoint.RIGHT_WRIST];
    
    let leftShoulder = this.model.boneDict['Left shoulder'];
    let rightShoulder = this.model.boneDict['Right shoulder'];
    
    let leftArm = this.model.boneDict['Left arm'];
    let rightArm = this.model.boneDict['Right arm'];
    
    let leftElbow = this.model.boneDict['Left elbow'];
    let rightElbow = this.model.boneDict['Right elbow'];
    
    // Relative to shoulder
    let leftElbowIdleVector = new THREE.Vector3(
      Math.cos(Math.PI / 4 * 7), 
      -Math.sin(Math.PI / 4 * 7),
      0
    );
    
    let leftElbow3DRelativeVector = new THREE.Vector3(
      leftElbow3DPose.x - leftShoulder3DPose.x,
      leftElbow3DPose.y - leftShoulder3DPose.y,
      leftElbow3DPose.z! - leftShoulder3DPose.z!,
    );
    
    
    let init = leftElbowIdleVector;
    let target = leftElbow3DRelativeVector;
    // let init = new THREE.Vector3(0.5, -0.5, 0);
    // let target = new THREE.Vector3(0.5, 0.5, 0.5);
    
    let axis = crossProduct(init, target);
    let axisNorm = axis.normalize();
    
    let dot = init.dot(target);
    let cosine = dot / (init.length() * target.length());
    let theta = Math.acos(cosine);
    let skew = getSkewSymmetricMatrix(axisNorm);
    let R = getRotationMatrix(theta, skew);
    
    let [alpha, beta, gamma] = rotationMatrixToEulerAnglesNew(R);
    
    leftArm.rotation.x = this.smoothMovement(alpha, leftArm.rotation.x, 0.1);
    leftArm.rotation.y = this.smoothMovement(-beta, leftArm.rotation.y, 0.1);
    leftArm.rotation.z = this.smoothMovement(-gamma, leftArm.rotation.z, 0.1);
    
    showXValue(alpha);
    showYValue(beta);
    showZValue(gamma);
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
    
    // console.log(origin.x, origin.y);
    
    let upperBody = this.model.boneDict['Upper body'];
    let head = this.model.boneDict['Head'];
    
    // Rotate upper body left and right
    // upperBody.rotation.z += this.shiftCoordinateNew(upperBody.rotation.z, origin.x);
    
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