import * as THREE from 'three';

import { GUI } from 'three/examples/jsm/libs/lil-gui.module.min.js';
import { MapControls } from 'three/examples/jsm/Addons.js';
import { MMDAnimationHelper } from 'three/examples/jsm/Addons.js';
import { MMDLoader } from 'three/addons/loaders/MMDLoader.js';
import { OutlineEffect } from 'three/examples/jsm/Addons.js';
import { Face } from '@tensorflow-models/face-landmarks-detection';

import { boneTranslations, morphTranslations } from './translations';
import { 
	createFaceLandmarksDetector,
	distance,
	drawFaceLandmarks, 
	getLandmarkCoordinate, 
	LandmarkPoint
} from './face_landmarks';



let mesh: THREE.SkinnedMesh;
let helper: MMDAnimationHelper;
let mouseX: number;
let mouseY: number;


// Clock
const clock = new THREE.Clock();

// Scene
let scene = new THREE.Scene();
let gridHelper = new THREE.PolarGridHelper( 30, 0 );
scene.add( gridHelper );

let ambient = new THREE.AmbientLight( 0xaaaaaa, 3 );
scene.add( ambient );

let directionalLight = new THREE.DirectionalLight( 0xffffff, 3 );
directionalLight.position.set( - 1, 1, 1 ).normalize();
scene.add( directionalLight );

// Camera
let camera = new THREE.PerspectiveCamera(
  75, window.innerWidth / window.innerHeight, 0.1, 1000
);

camera.position.y = 19;
camera.position.z = 10;
camera.rotation.x += -10 * Math.PI / 180;

// Renderer
let renderElement = document.getElementById('main-render');
let renderer = new THREE.WebGLRenderer( {antialias: true} );
renderer.setSize(window.innerWidth, window.innerHeight);
renderElement!.appendChild(renderer.domElement);

// Effects
let effects = new OutlineEffect(renderer);

// Camera Controls
// Caveat: This will reset the camera rotation.
// let controls = new MapControls(camera, renderer.domElement);

// Background color
scene.background = new THREE.Color(0xffffff);



function saveFileAndDownload(filename: string, content: string) {
	let link = window.document.createElement('a');
	let blob = new Blob([content], {type: 'string'});
	link.href = window.URL.createObjectURL(blob);
	link.download = filename;
	link.click();
}

function getModelBones(mesh) {
	let bones: any[] = [];
	
	if (!mesh.children.length) {
		return bones;
	}
	
	for (let child of mesh.children) {
		if (child.isBone) {
			bones.push(child);
			let childBones = getModelBones(child);
			bones.push(...childBones);
		}
	}
	return bones;
}

function createGUI(mesh: THREE.SkinnedMesh) {
	let gui = new GUI();
	let controls = {};
	let morphsGUI = gui.addFolder('morphs');
	let morphDict = mesh.morphTargetDictionary;
	let bonesGUI = gui.addFolder('bones');
	let bones: THREE.Bone[] = mesh.skeleton.bones;
	let boneNames: string[] = [];
	let boneDict = {};
	
	
	// Model morphs
	for (let key in morphDict) {
		let morphNum = morphDict[key];
		let engName = morphTranslations[key] ?? key;
		controls[engName] = 0;
		
		//@ts-expect-error
		morphsGUI.add(controls, engName, 0.0, 1.0, 0.01).onChange((value) => {
			mesh.morphTargetInfluences![morphNum] = value;
		});
	}
	
	// Model bones
	for (let bone of bones) {
		let engName = boneTranslations[bone.name];
		boneDict[engName] = bone;
	}
	
	for (let key in boneDict) {
		// if (!key.includes('eye')) continue;
		
		let currentBone: THREE.Bone = boneDict[key];
		controls[key] = currentBone.rotation.z;
		
		let maxValue = 8;
		//@ts-expect-error
		bonesGUI.add(controls, key, -maxValue, maxValue, 0.01).onChange((value) => {
			currentBone.rotation.z = value;
		})
	}
	
	morphsGUI.close();
	
	return {gui, boneDict, morphDict};
}

// Model Texture
// let texture = new THREE.TextureLoader().load('./assets/alpha/textures/Body.png');
// let material = new THREE.MeshBasicMaterial({ map: texture });

// let bones: any[] = [];
// function traverseModel(child) {
// 	if (child.isMesh) {
// 		// child.material = material;		
// 	}
// 	if (child.isBone) {
// 		// console.log(child.name, child.material);
// 		console.log(child);
// 		bones.push(child);
// 	}
// 	// console.log(child.constructor.name);
// }

function clamp(val, min, max) {
	return val > max ? max : val < min ? min : val;
}

function trackMouseCoordinate() {
	let coordinateX = document.getElementById('x-coordinate');
	let coordinateY = document.getElementById('y-coordinate');
	
	let factorX = window.innerWidth / 2;
	let factorY = window.innerHeight / 2;
	
	document.body.onmousemove = (event) => {
		event.preventDefault();
		// Shift the origin from top-left to center
		mouseX = (event.x - factorX) / factorX;
		mouseY = (event.y - factorY) / factorY;
		coordinateX!.innerHTML = mouseX.toFixed(2);
		coordinateY!.innerHTML = mouseY.toFixed(2);
	}
}

async function showCamera() {
	let video = document.getElementById("video") as HTMLVideoElement;
  if (navigator.mediaDevices) {
    const stream = await navigator.mediaDevices.getUserMedia({ "video": true });
    video.srcObject = stream;
  }
	return video;
}


function moveModel(
	mesh: THREE.SkinnedMesh,
	faces: Face[], 
	boneDict: Record<string, THREE.Bone> | null
) {
	
	if (!boneDict) {
		return;
	}
	
	let noseMiddle = getLandmarkCoordinate(faces, LandmarkPoint.NOSE_MIDDLE);
	let leftEyelidTop = getLandmarkCoordinate(faces, LandmarkPoint.LEFT_EYELID_TOP);
	let leftEyelidBottom = getLandmarkCoordinate(faces, LandmarkPoint.LEFT_EYELID_BOTTOM);
	let rightEyelidTop = getLandmarkCoordinate(faces, LandmarkPoint.RIGHT_EYELID_TOP);
	let rightEyelidBottom = getLandmarkCoordinate(faces, LandmarkPoint.RIGHT_EYELID_BOTTOM);
	
	// Eyes blink
	let leftEyelidDelta = distance(
		[leftEyelidTop.x, leftEyelidTop.y], 
		[leftEyelidBottom.x, leftEyelidBottom.y]
	);
	
	let rightEyelidDelta = distance(
		[rightEyelidTop.x, rightEyelidTop.y], 
		[rightEyelidBottom.x, rightEyelidBottom.y]
	);
	
	if (rightEyelidDelta < 0.1) {
		
	}
	
	
	// boneDict['Right eye'].rotation.x = clamp(mouseY, -0.10, 0.30);
	// boneDict['Right eye'].rotation.y = clamp(mouseX, -0.20, 0.10);
	// boneDict['Left eye'].rotation.x = clamp(mouseY, -0.10, 0.30);
	// boneDict['Left eye'].rotation.y = clamp(mouseX, -0.20, 0.10);
	// boneDict['Head'].rotation.x = mouseY;
	boneDict['Head'].rotation.x = noseMiddle.y!;
	// boneDict['Head'].rotation.y = noseMiddle.z!;
	boneDict['Head'].rotation.z = noseMiddle.x!;
	
	boneDict['Upper body'].rotation.z = noseMiddle.x!;
}


// Render Scene
async function animate(
	boneDict: Record<string, THREE.Bone> | null,
	// morphDict: Record<string, number>,
) {
	
	let video = await showCamera();
	let canvas = document.getElementById('video-face-landmark') as HTMLCanvasElement;
	let ctx = canvas.getContext('2d')!;
	
	const detector = await createFaceLandmarksDetector();
	
	async function loop() {
		requestAnimationFrame( loop );
		
		// Estimate and draw face landmarks
		let faces = await detector.estimateFaces(video, {flipHorizontal: true});
		if (!faces.length) return;
		
		let noseMiddle = getLandmarkCoordinate(faces, LandmarkPoint.NOSE_MIDDLE);
		// ctx.clearRect(0, 0, canvas.width, canvas.height);
		// drawFaceLandmarks(ctx, faces);
		
		if (boneDict) {
			// boneDict['Right eye'].rotation.x = clamp(mouseY, -0.10, 0.30);
			// boneDict['Right eye'].rotation.y = clamp(mouseX, -0.20, 0.10);
			// boneDict['Left eye'].rotation.x = clamp(mouseY, -0.10, 0.30);
			// boneDict['Left eye'].rotation.y = clamp(mouseX, -0.20, 0.10);
			// boneDict['Head'].rotation.x = mouseY;
			boneDict['Head'].rotation.x = noseMiddle.y!;
			// boneDict['Head'].rotation.y = noseMiddle.z!;
			boneDict['Head'].rotation.z = noseMiddle.x!;
			
			boneDict['Upper body'].rotation.z = noseMiddle.x!;
			
			
		}
		
		
		helper.update(clock.getDelta());
		effects.render( scene, camera );
	}
	loop();
}


async function main() {
	
	trackMouseCoordinate();
	
	// Load MMD Model with Animation
	// vmdPath must exist or else the model won't load
	let modelPath = 'assets/miku/miku_v2.pmd';
	let vmdPath = 'assets/miku/wavefile_v2.vmd';
	let mmdLoader = new MMDLoader();
	helper = new MMDAnimationHelper({
		afterglow: 0.0,
	});
	
	mmdLoader.loadWithAnimation(modelPath, vmdPath, (mmd) => {
		// Add 3D Model to scene
		mesh = mmd.mesh;
		scene.add(mesh);
		
		let {gui, boneDict, morphDict} = createGUI(mesh);
		
		// Handle MMD Animations
		//
		// Caveat: without ammo js, adding mesh into this helper
		//				 will block the entire script.
		// let animationHelper = helper.add(mesh, {animation: mmd.animation});
		
		// Inverse kinematics
		// let ikHelper = helper.objects.get(mesh)!.ikSolver.createHelper();
		// ikHelper.visible = false;
		// scene.add(ikHelper);
		
		// Physics
		// let physicsHelper = helper.objects.get(mesh)!.physics!.createHelper();
		// physicsHelper.visible = false;
		// scene.add(physicsHelper);
		
		// helper.enable('animation', false);
		animate(boneDict);
	})
	
}

// Use ammo.js from CDN
//@ts-expect-error
Ammo().then((AmmoLib) => {
	
	//@ts-expect-error
	Ammo = AmmoLib;
	main();
})
