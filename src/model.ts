import * as THREE from 'three';
import { MMDLoader } from 'three/addons/loaders/MMDLoader.js';

import { boneTranslations, morphTranslations } from './translations';



function saveFileAndDownload(filename: string, content: string) {
	let link = window.document.createElement('a');
	let blob = new Blob([content], {type: 'string'});
	link.href = window.URL.createObjectURL(blob);
	link.download = filename;
	link.click();
}


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
    // let boneNames: string[] = [];
    for (let bone of bones) {
      let boneName = bone.name;
      // boneNames.push(boneName);
      let boneEngName = boneTranslations[boneName] ?? boneName;
      this.boneDict[boneEngName] = bone;
    }
    // saveFileAndDownload('bones.txt', boneNames.join('\n'));
    // console.log('Bone total: ', bones.length);
    
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