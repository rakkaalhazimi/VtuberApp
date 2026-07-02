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
    directionalLight.position.set(3, 8, 5);
    directionalLight.castShadow = true;
    scene.add( directionalLight );
    
    const hemi = new THREE.HemisphereLight(
      0xffffff, // sky
      0x404040, // ground
      1.2
    );
    scene.add(hemi);
    
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
    camera.position.z = 10;
    camera.rotation.x += -10 * Math.PI / 180;
  
    // Background color
    scene.background = new THREE.Color(0x111827);
    
    // // Track mouse movement
    // await this.trackMouseCoordinate();
    
    // Load model
    let model = new Model();
    // let mikuModel = await model.loadModel('assets/miku/miku_v2.pmd');
    // let mikuModel = await model.loadModel('assets/alpha/alpha.pmx');
    // let mikuModel = await model.loadModel('assets/female_rover/Female Rover1.01.pmx');
    let mikuModel = await model.loadModel('assets/feixiao/星穹铁道—飞霄.pmx');
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
    let leftShoulder = model.boneDict['Left shoulder'];
    let leftElbow = model.boneDict['Left elbow'];
    let leftArm = model.boneDict['Left arm'];
    let leftWrist = model.boneDict['Left wrist'];
    // let rightArm = model.boneDict['Right arm'];
    
    // let bind = leftWrist.position.clone().sub(leftElbow.position).normalize();
    // let bindRotation = new THREE.Quaternion().setFromUnitVectors(
    //   new THREE.Vector3(0, -1, 0),
    //   bind
    // );
    
    // console.log('Bind pose: ', bind);
    
    // const from = new THREE.Vector3(
    //   Math.cos(7/4 * Math.PI), 
    //   Math.sin(7/4 * Math.PI), 
    //   0
    // ).normalize();
    // const to = new THREE.Vector3(
    //   Math.cos(0), 
    //   Math.sin(0), 
    //   -Math.cos(0),
    // );
    // const q = new THREE.Quaternion();
    // q.setFromUnitVectors(bind, to);
    // leftArm.applyQuaternion(bindRotation.invert());
    
    
    // leftArm.rotation.z = -0.5;
    // rightArm.rotation.z = 0.5;
    
    // Hands up
    // leftArm.rotation.x = -1.03;
    // leftArm.rotation.y = -1.24;
    // leftArm.rotation.z = 1.16;
    
    // Bones
    // let bones: any[] = [];
    // for (let key in model.boneDict) {
    //   bones.push(model.boneDict[key]);
    // }
    // console.log("Bones: ", bones);
    
    // Draw sphere on left arm bone
    // let geometry = new THREE.SphereGeometry( 1, 32, 16 ); 
    // let material = new THREE.MeshBasicMaterial( { color: 0xffff00 } ); 
    // let sphere = new THREE.Mesh( geometry, material );
    // let position = new THREE.Vector3();
    // rightArm.getWorldPosition(position);
    // console.log(position.x, position.y, position.z);
    // sphere.position.x = position.x;
    // sphere.position.y = position.y;
    // sphere.position.z = position.z;
    // scene.add( sphere );
    
    // let boneSphereMap = new Map();
    // boneSphereMap.set(sphere.id, rightArm.id);
    
    // let testJoint = new DummyJoints(15, 20);
    // testJoint.rotateShoulder(0, 0, degToRad(45));
    // testJoint.rotateElbow(0, 0, 0);
    // scene.add(testJoint.group);
    
    // leftArm.rotation.set(0, 0, degToRad(45));
    
    
    
    // Raycasting
    let raycaster = new THREE.Raycaster();
    let mouse = new THREE.Vector2();
    
    // window.addEventListener('click', (event) => {
    //   // Calculate mouse position in normalized device coordinates (-1 to +1)
    //   mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    //   mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
  
    //   raycaster.setFromCamera(mouse, camera);
    //   console.log(mouse.x, mouse.y);
  
    //   // Check for intersections with the bones
    //   // const intersects = raycaster.intersectObjects(bones, true);
    //   const intersects = raycaster.intersectObjects([sphere]);
    //   console.log("Intersects: ", intersects);
  
    //   if (intersects.length > 0) {
    //       const boneId = boneSphereMap.get(intersects[0].object.id);
    //       const selectedBone = scene.getObjectById(boneId)!;
    //       // Rotate or manipulate the selected bone
    //       selectedBone.rotation.x += Math.PI / 4; // example rotation
    //   }
    // });
    
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
    
    let poseEstimation = new PoseEstimation(video.height, video.width);
    // await poseEstimation.loadMovenetPoseEstimationDetector();
    await poseEstimation.loadBlazePoseEstimationDetector();
    
    let moveGuider = new ModelMovementGuider(model, faceLandmark, poseEstimation);
    
    async function animate() {
      // requestAnimationFrame(animate);
      
      setTimeout( function() {
        requestAnimationFrame( animate );
      }, 1000 / 25 );
      
      
      let faces = await faceLandmark.estimateFaces(video);
      let poses = await poseEstimation.estimatePose(video);
      
      // Draw face landmarks
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      faceLandmark.drawFaceLandmarks(canvasCtx, faces);
      poseEstimation.drawPoseLandmarks(canvasCtx, poses);
      
      moveGuider.guideHeadRotation(faces);
      moveGuider.guideBlinking(faces);
      moveGuider.guideMouthMovement(faces);
      moveGuider.guideUpperBodyMovement(poses);
      moveGuider.guideLeftArmMovement(poses);
      moveGuider.guideRightArmMovement(poses);
      
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