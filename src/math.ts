// Compute the distance between two 2D vectors.
export function euclideanDistance2D(a: number[], b: number[]) {
  return Math.sqrt(
    Math.pow(a[0] - b[0], 2) + 
    Math.pow(a[1] - b[1], 2)
  );
}

export function gradient2D(a: number[], b: number[]) {
  let deltaX = b[0] - a[0];
  let deltaY = b[1] - a[1];
  
  return deltaY / deltaX;
}

// Compute the angle of B in ABC line
export function angleOfTriangle2D(a: number [], b: number[], c: number[], includeObscute: boolean = false) {
  
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
  
  let dotProduct = (vectorBA.x * vectorBC.x) + (vectorBA.y + vectorBC.y);
  let cosine = dotProduct / (lengthAB * lengthCB);
  // Math.acos can only receive value between -1 and 1
  cosine = Math.max(-1, Math.min(1, cosine));
  let radian = Math.acos(cosine);
  
  // Determine acute or obscute angle with cross product
  let crossProduct = (vectorBA.x * vectorBC.y) - (vectorBA.y * vectorBC.x);
  let sign = Math.sign(crossProduct);
  
  // If positive sign, keep the radian.
  // If negative sign, subtract the radian with 2 * pi.
  if (sign < 0 && includeObscute) {
    radian = (2 * Math.PI) - radian;
  }
  
  return radian;
}