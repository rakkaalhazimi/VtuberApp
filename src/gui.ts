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
    console.log(controls);
  }
  
}