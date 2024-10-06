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
export function angleOfTriangle2D(a: number [], b: number[], c: number[]) {
  
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
  
  let vectorCA = {
    x: c[0] - b[0],
    y: c[1] - b[1]
  }
  
  let dotProduct = (vectorBA.x * vectorCA.x) + (vectorBA.y + vectorCA.y);
  let cosine = dotProduct / (lengthAB * lengthCB);
  let radian = Math.acos(cosine);
  return radian;
}