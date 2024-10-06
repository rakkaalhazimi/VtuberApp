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