import * as THREE from 'three';

import { MMDAnimationHelper, MapControls, OrbitControls } from 'three/examples/jsm/Addons.js';
import { OutlineEffect } from 'three/examples/jsm/Addons.js';

import { FaceLandmark } from './face-landmarks';
import { AppGUI } from './gui';
import { ModelMovementGuider } from './guider';
import { Model } from './model';
import { PoseEstimation } from './pose-estimation';



class App {
  
  mouseX: number = 0;
  mouseY: number = 0;
  
  async trackMouseCoordinate() {
    let coordinateX = document.getElementById('x-coordinate');
    let coordinateY = document.getElementById('y-coordinate');
    
    let factorX = window.innerWidth / 2;
    let factorY = window.innerHeight / 2;
    
    document.body.onmousemove = (event) => {
      event.preventDefault();
      // Shift the origin from top-left to center
      this.mouseX = (event.x - factorX) / factorX;
      this.mouseY = (event.y - factorY) / factorY;
      coordinateX!.innerHTML = this.mouseX.toFixed(2);
      coordinateY!.innerHTML = this.mouseY.toFixed(2);
    }
  }
  
  async showCamera(): Promise<HTMLVideoElement> {
    let video = document.getElementById("video") as HTMLVideoElement;
    if (navigator.mediaDevices) {
      const stream = await navigator.mediaDevices.getUserMedia({ "video": true });
      video.srcObject = stream;
    }
    return video;
  }
  
  async createCanvas(): 
    Promise<[
      canvas: HTMLCanvasElement, 
      ctx: CanvasRenderingContext2D
  ]> {
    let canvas = document.getElementById('video-face-landmark') as HTMLCanvasElement;
	  let ctx = canvas.getContext('2d')!;
    return [canvas, ctx];
  }
  
  async start() {
    // Clock
    let clock = new THREE.Clock();
  
    // Scene
    let scene = new THREE.Scene();
    let gridHelper = new THREE.PolarGridHelper( 30, 0 );
    scene.add( gridHelper );
  
    let ambient = new THREE.AmbientLight( 0xaaaaaa, 2 );
    scene.add( ambient );
  
    let directionalLight = new THREE.DirectionalLight( 0xffffff, 2 );
    directionalLight.position.set( 0, 0, 0 ).normalize();
    scene.add( directionalLight );
  
    // Camera
    let camera = new THREE.PerspectiveCamera(
      75, window.innerWidth / window.innerHeight, 0.1, 1000
    );
    
  
    // Renderer
    let renderElement = document.getElementById('main-render');
    let renderer = new THREE.WebGLRenderer( {antialias: true} );
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderElement!.appendChild(renderer.domElement);
  
    // Effects
    let effects = new OutlineEffect(renderer);
  
    // Camera Controls
    // Caveat: This will reset the camera rotation.
    //         so we need to change the camera position afterward.
    // let controls = new MapControls(camera, renderer.domElement);
    let controls = new OrbitControls(camera, renderer.domElement);
    camera.position.y = 19;
    camera.position.z = 5;
    camera.rotation.x += -10 * Math.PI / 180;
  
    // Background color
    scene.background = new THREE.Color(0x111827);
    
    
    // Load model
    let model = new Model();
    let mikuModel = await model.loadModel('assets/miku/miku_v2.pmd');
    scene.add(mikuModel);
    
    // Model animation helper
    let helper = new MMDAnimationHelper({afterglow: 0.0});
    let animationHelper = helper.add(mikuModel);
    
    // Inverse Kinematic
    let ikHelper = 
      helper.objects.get(mikuModel)!.ikSolver.createHelper();
    ikHelper.visible = false;
    scene.add(ikHelper);
    
    // Physics
    let physicsHelper = 
      helper.objects.get(mikuModel)!.physics!.createHelper();
		physicsHelper.visible = false;
		scene.add(physicsHelper);
    
    
    // Make model arms rotate downward
    let leftArm = model.boneDict['Left arm'];
    let rightArm = model.boneDict['Right arm'];
    leftArm.rotation.z = -0.5;
    rightArm.rotation.z = 0.5;
    
    // GUI
    let gui = new AppGUI();
    gui.createMorphGUI(model);
    gui.createBoneGUI(model);
    model.morph('Grin', 1);
    
    // Canvas
    let [canvas, canvasCtx] = await this.createCanvas();
    
    // Stuffs to guide the movement of the model
    let video = await this.showCamera();
    let faceLandmark = new FaceLandmark(video.width, video.height);
    await faceLandmark.loadFaceLandmarksDetector();
    let moveGuider = new ModelMovementGuider(model, faceLandmark);
    
    let poseEstimation = new PoseEstimation(video.height, video.width);
    await poseEstimation.loadMovenetPoseEstimationDetector();
    // await poseEstimation.loadBlazePoseEstimationDetector();
    
    async function animate() {
      requestAnimationFrame(animate);
      
      let faces = await faceLandmark.estimateFaces(video);
      
      // Draw face landmarks
      // canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      // faceLandmark.drawFaceLandmarks(canvasCtx, faces);
      
      moveGuider.guideMovement(faces);
      
      // let upperBody = model.boneDict['Upper body'];
      // upperBody.rotation.z += 0.1;
      
      // This will make the physics and ik works
      helper.update(clock.getDelta());
      
      effects.render(scene, camera);
    }
    animate();
    
  }
  
}

// Use ammo.js from CDN
//@ts-expect-error
Ammo().then((AmmoLib) => {
	
	//@ts-expect-error
	Ammo = AmmoLib;
  let app = new App();
  app.start();
});