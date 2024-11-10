import * as THREE from 'three';


// Compute the distance between two 2D vectors.
export function euclideanDistance2D(a: number[], b: number[]) {
  return Math.sqrt(
    Math.pow(a[0] - b[0], 2) +
    Math.pow(a[1] - b[1], 2)
  );
}

export function euclideanDistance3D(a: number[], b: number[]) {
  return Math.sqrt(
    Math.pow(a[0] - b[0], 2) +
    Math.pow(a[1] - b[1], 2) +
    Math.pow(a[2] - b[2], 2)
  );
}

export function gradient2D(a: number[], b: number[]) {
  let deltaX = b[0] - a[0];
  let deltaY = b[1] - a[1];

  return deltaY / deltaX;
}

// Compute the angle of B in ABC line
export function angleOfTriangle2D(a: number[], b: number[], c: number[], includeObscute: boolean = false) {

  let lengthAB = euclideanDistance2D(
    [a[0], a[1]],
    [b[0], b[1]]
  );

  let lengthCB = euclideanDistance2D(
    [c[0], c[1]],
    [b[0], b[1]]
  );

  let vectorBA = {
    x: a[0] - b[0],
    y: a[1] - b[1]
  };

  let vectorBC = {
    x: c[0] - b[0],
    y: c[1] - b[1]
  }

  let dotProduct = (vectorBA.x * vectorBC.x) + (vectorBA.y * vectorBC.y);
  let cosine = dotProduct / (lengthAB * lengthCB);
  // Math.acos can only receive value between -1 and 1
  cosine = Math.max(-1, Math.min(1, cosine));
  let radian = Math.acos(cosine);

  // Determine acute or obscute angle with cross product
  let crossProduct = (vectorBA.x * vectorBC.y) - (vectorBA.y * vectorBC.x);
  let sign = Math.sign(crossProduct);

  // Include Obscute angle
  // If positive sign, keep the radian.
  // If negative sign, subtract the radian with 2 * pi.
  if (sign < 0 && includeObscute) {
    radian = (2 * Math.PI) - radian;
  }

  return radian;
}

export function rotateVector2D(a: number[], radian: number) {
  let [x, y] = a;
  return {
    x: x * Math.cos(radian) - y * Math.sin(radian),
    y: x * Math.sin(radian) + y * Math.cos(radian)
  };
}

export function crossProduct(a: THREE.Vector3, b: THREE.Vector3) {
  let c = new THREE.Vector3(
    (a.y * b.z) - (a.z * b.y),
    (a.z * b.x) - (a.x * b.z),
    (a.x * b.y) - (a.y * b.x),
  );
  return c;
}

export function getSkewSymmetricMatrix(a: THREE.Vector3) {
  let [x, y, z] = [a.x, a.y, a.z];
  let skew = new THREE.Matrix3();
  skew.elements = [0, -z, y, z, 0, -x, -y, x, 0];
  return skew;
}

export function matrixMultiplication(m1: number[][], m2: number[][]) {
  let result: number[][] = [];
  for (let i = 0; i < m1.length; i++) {
    result[i] = [];
    for (let j = 0; j < m2[0].length; j++) {
      let sum = 0;
      for (let k = 0; k < m1[0].length; k++) {
        sum += m1[i][k] * m2[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

export function getRotationMatrix(theta: number, skew: THREE.Matrix3) {
  let I = new THREE.Matrix3();
  I.set(
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
  );
  
  // sin(theta) * skew
  let sinTheta = new THREE.Matrix3();
  sinTheta.elements = Array(9).fill(Math.sin(theta));
  for (let i = 0; i < 9; i++) {
    sinTheta.elements[i] *= skew.elements[i];
  }
  
  // (1 - cos(theta)) * skew^2
  let cosTheta = new THREE.Matrix3();
  cosTheta.elements = Array(9).fill(1 - Math.cos(theta));
  let skewSquares = new THREE.Matrix3().multiplyMatrices(skew, skew);
  for (let i = 0; i < 9; i++) {
    cosTheta.elements[i] *= skewSquares.elements[i];
  }
  
  // I + sin(theta) * skew + (1 - cos(theta)) * skew^2
  let R = new THREE.Matrix3();
  for (let i = 0; i < 9; i++) {
    R.elements[i] = I.elements[i] + sinTheta.elements[i] + cosTheta.elements[i];
  }
  
  return R;
}

export function rotationMatrixToEulerAngles(R: number[][]) {
  // Compute sy (sin(y) and cos(y)) to check for gimbal lock
  const sy = Math.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0]);

  let singular = sy < 1e-6; // If sy is close to zero, we have a gimbal lock

  let alpha, beta, gamma;

  if (!singular) {
    // Normal case: No gimbal lock
    alpha = Math.atan2(R[2][1], R[2][2]);  // Rotation around X-axis
    beta = Math.atan2(-R[2][0], sy);       // Rotation around Y-axis
    gamma = Math.atan2(R[1][0], R[0][0]);  // Rotation around Z-axis
  } else {
    // Gimbal lock: Set alpha to 0 and calculate beta and gamma differently
    alpha = 0;
    beta = Math.atan2(-R[2][0], sy);
    gamma = Math.atan2(-R[1][2], R[1][1]);
  }

  return [alpha, beta, gamma];
}

export function rotationMatrixToEulerAnglesNew(R: THREE.Matrix3) {
  // Compute sy (sin(y) and cos(y)) to check for gimbal lock
  const sy = Math.sqrt(R.elements[0] * R.elements[0] + R.elements[3] * R.elements[3]);

  let singular = sy < 1e-6; // If sy is close to zero, we have a gimbal lock

  let alpha, beta, gamma;

  if (!singular) {
    // Normal case: No gimbal lock
    alpha = Math.atan2(R.elements[7], R.elements[8]);   // Rotation around X-axis
    beta = Math.atan2(-R.elements[6], sy);              // Rotation around Y-axis
    gamma = Math.atan2(R.elements[3], R.elements[0]);   // Rotation around Z-axis
  } else {
    // Gimbal lock: Set alpha to 0 and calculate beta and gamma differently
    alpha = 0;
    beta = Math.atan2(-R.elements[7], sy);
    gamma = Math.atan2(-R.elements[5], R.elements[4]);
  }

  return [alpha, beta, gamma];
}

export function eulerAnglesFromVectorMovement(init: THREE.Vector3, target: THREE.Vector3) {
  let axis = crossProduct(init, target);
  let axisNorm = axis.normalize();
  
  let dot = init.dot(target);
  let cosine = dot / (init.length() * target.length());
  let theta = Math.acos(cosine);
  let skew = getSkewSymmetricMatrix(axisNorm);
  let R = getRotationMatrix(theta, skew);
  
  let [alpha, beta, gamma] = rotationMatrixToEulerAnglesNew(R);
  
  return [alpha, beta, gamma];
}