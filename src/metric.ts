let x = document.getElementById('x-coordinate');
let y = document.getElementById('y-coordinate');
let z = document.getElementById('z-coordinate');
    
export function showXValue(value: number) {
  x!.innerHTML = value.toFixed(3);
}

export function showYValue(value: number) {
  y!.innerHTML = value.toFixed(3);
}

export function showZValue(value: number) {
  z!.innerHTML = value.toFixed(3);
}