import * as THREE from 'three';
import { MMDAnimationHelper } from 'three/examples/jsm/Addons.js';
import { MMDLoader } from 'three/addons/loaders/MMDLoader.js';
import { Face, Keypoint } from '@tensorflow-models/face-landmarks-detection';

import { FaceLandmark, LandmarkPoint } from './face-landmarks';
import { boneTranslations, morphTranslations } from './translations';



const MIN_EAR_THRES = 0.2;
const MAX_MAR_THRES = 0.3;


export class Model {
  public mesh: THREE.SkinnedMesh;
  public loader: MMDLoader;
  public boneDict: Record<string, THREE.Bone> = {};
  public morphDict: Record<string, number> = {};
  
  constructor() {
    this.loader = new MMDLoader();
  }
  
  async loadModel(modelPath: string) {
    // Get model 3D object
    this.mesh = await this.loader.loadAsync(modelPath);
    
    // Map model morphs with its english names
    let morphTargetDict = this.mesh.morphTargetDictionary!;
    for (let morphName in morphTargetDict) {
		  let morphEngName = morphTranslations[morphName] ?? morphName;
      let morphNum = morphTargetDict[morphName];
      this.morphDict[morphEngName] = morphNum;
    }
    
    // Map model bone with its english names
    let bones = this.mesh.skeleton.bones;
    for (let bone of bones) {
      let boneName = bone.name;
      let boneEngName = boneTranslations[boneName];
      this.boneDict[boneEngName] = bone;
    }
    
    return this.mesh;
  }
  
  getMorphValue(morphName: string) {
    let morphNum = this.morphDict[morphName];
    let morphValue = this.mesh.morphTargetInfluences![morphNum];
    return morphValue;
  }
  
  morph(morphName: string, value: number) {
    let morphNum = this.morphDict[morphName];
    this.mesh.morphTargetInfluences![morphNum] = value;
  }
  
}


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
    speed: number = 0.01,
    multiplier: number = 1,
  ) {
    
    let diff = Math.abs(origin - target * multiplier);
    if (diff <= 0.02) {
      return 0;
    }
    if (origin < target * multiplier) {
      return speed;
    } else {
      return -speed;
    }
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
  
  guideOpenMouth(faces: Face[]) {
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
    
    // let x = document.getElementById('x-coordinate');
    // let y = document.getElementById('y-coordinate');
    // let z = document.getElementById('z-coordinate');
    // x!.innerHTML = mar.toFixed(3);
    // y!.innerHTML = rightEAR.toFixed(3);
    // z!.innerHTML = (leftEyeAspectRatio).toFixed(3);
    // z!.innerHTML = leftEAR.toFixed(3);
    
    let mouthOpenValue = this.model.getMorphValue('A');
    let valueShift: number;
    
    // Open mouth
    if (mar > MAX_MAR_THRES) {
      valueShift = this.shiftCoordinate(mouthOpenValue, 0.3, 0.02);
    // Close mouth
    } else {
      valueShift = this.shiftCoordinate(mouthOpenValue, 0, 0.02);
    }
    this.model.morph('A', valueShift);
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
    head.rotation.z += this.shiftCoordinateNew(head.rotation.z, roll);
    head.rotation.y += this.shiftCoordinateNew(head.rotation.y, -yaw);
    head.rotation.x += this.shiftCoordinateNew(head.rotation.x, pitch);
    
    // let x = document.getElementById('x-coordinate');
    // let y = document.getElementById('y-coordinate');
    // let z = document.getElementById('z-coordinate');
    // x!.innerHTML = roll.toFixed(3);
    // y!.innerHTML = yaw.toFixed(3);
    // z!.innerHTML = pitch.toFixed(3);
    
  }
  
  guideUpperBodyMovement(faces: Face[]) {
    let origin = 
      this.faceLandmark.getScaledLandmarkCoordinate(
        faces, LandmarkPoint.NOSE_MIDDLE
    );
    
    let upperBody = this.model.boneDict['Upper body'];
    let head = this.model.boneDict['Head'];
    
    // Rotate upper body left and right
    upperBody.rotation.z += this.shiftCoordinateNew(upperBody.rotation.z, origin.x);
    
    // Rotate upper body forward and backward;
    upperBody.rotation.x += this.shiftCoordinateNew(upperBody.rotation.x, origin.z!);
    
    // Rotate head upward if we get too close
    head.rotation.x += this.shiftCoordinateNew(head.rotation.x, -origin.z!);
    
    // Align head straight as the upper body rotate left and right
    head.rotation.z += this.shiftCoordinateNew(head.rotation.z, -origin.x);
  }
  
  guideMovement(faces: Face[]) {
    if (!faces.length) {
      return;
    }
    this.guideBlinking(faces);
    // this.guideOpenMouth(faces);
    this.guideHeadRotation(faces);
    this.guideUpperBodyMovement(faces);
  }
  
}