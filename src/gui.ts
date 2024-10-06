import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { Model } from './model';

export class AppGUI {
  
  public gui: GUI
  
  constructor() {
    this.gui = new GUI();
  }
  
  createMorphGUI(model: Model) {
    let morphGUI = this.gui.addFolder('morph');
    let controls = {}
    for (let key in model.morphDict) {
      controls[key] = 0;
      //@ts-expect-error
      morphGUI.add(controls, key, 0.0, 1.0, 0.01).onChange((value: number) => {
        model.morph(key, value);
      })
    }
    morphGUI.close();
    console.log(controls);
  }
  
  createBoneGUI(model: Model) {
    let boneGUI = this.gui.addFolder('bone');
    let controls = {}
    for (let key in model.boneDict) {
      
      if (key.includes('肩.L')) {
        continue;
      }
      
      let boneItemGUI = boneGUI.addFolder(key);
      
      let rotationXControl = `${key}-rotation-x`
      controls[rotationXControl] = model.boneDict[key].rotation.x;
      //@ts-expect-error
      boneItemGUI.add(controls, rotationXControl, -2.0, 2.0, 0.01).onChange((value: number) => {
        model.boneDict[key].rotation.x = value;
      });
      
      let rotationYControl = `${key}-rotation-y`
      controls[rotationYControl] = model.boneDict[key].rotation.y;
      //@ts-expect-error
      boneItemGUI.add(controls, rotationYControl, -2.0, 2.0, 0.01).onChange((value: number) => {
        model.boneDict[key].rotation.y = value;
      });
      
      let rotationZControl = `${key}-rotation-z`
      controls[rotationZControl] = model.boneDict[key].rotation.z;
      //@ts-expect-error
      boneItemGUI.add(controls, rotationZControl, -2.0, 2.0, 0.01).onChange((value: number) => {
        model.boneDict[key].rotation.z = value;
      });
      
    }
    console.log(controls);
  }
  
}