// Compute the distance between two 2D vectors.
export function euclideanDistance2D(a: number[], b: number[]) {
  return Math.sqrt(
    Math.pow(a[0] - b[0], 2) + 
    Math.pow(a[1] - b[1], 2)
  );
}