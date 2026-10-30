const imageFile = document.querySelector("#imageFile");
const uploadZone = document.querySelector("#uploadZone");
const uploadTitle = document.querySelector("#uploadTitle");
const uploadHint = document.querySelector("#uploadHint");
const cameraVideo = document.querySelector("#cameraVideo");
const cameraCanvas = document.querySelector("#cameraCanvas");
const cameraData = document.querySelector("#cameraData");
const cameraMessage = document.querySelector("#cameraMessage");
const startCamera = document.querySelector("#startCamera");
const capturePhoto = document.querySelector("#capturePhoto");
let cameraStream;

function stopCamera() {
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  cameraVideo.hidden = true;
  capturePhoto.hidden = true;
  startCamera.hidden = false;
}

imageFile.addEventListener("change", () => {
  cameraData.value = "";
  const file = imageFile.files[0];
  uploadTitle.textContent = file ? file.name : "Drop an image here";
  uploadHint.textContent = file ? "Ready to analyze" : "or browse JPG, JPEG, PNG";
  stopCamera();
});

startCamera.addEventListener("click", async () => {
  cameraMessage.textContent = "";
  if (!navigator.mediaDevices?.getUserMedia) {
    cameraMessage.textContent = "Camera access is not available in this browser.";
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    cameraVideo.srcObject = cameraStream;
    cameraVideo.hidden = false;
    capturePhoto.hidden = false;
    startCamera.hidden = true;
    cameraData.value = "";
  } catch (_error) {
    cameraMessage.textContent = "Allow camera access or choose an image file instead.";
  }
});

capturePhoto.addEventListener("click", () => {
  cameraCanvas.width = cameraVideo.videoWidth;
  cameraCanvas.height = cameraVideo.videoHeight;
  cameraCanvas.getContext("2d").drawImage(cameraVideo, 0, 0);
  cameraData.value = cameraCanvas.toDataURL("image/jpeg", 0.9);
  uploadTitle.textContent = "Photo captured";
  uploadHint.textContent = "Ready to analyze";
  imageFile.value = "";
  stopCamera();
});

for (const eventName of ["dragenter", "dragover"]) {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.add("is-dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.remove("is-dragging");
  });
}

uploadZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file && ["image/jpeg", "image/png"].includes(file.type)) {
    imageFile.files = event.dataTransfer.files;
    imageFile.dispatchEvent(new Event("change", { bubbles: true }));
  } else {
    cameraMessage.textContent = "Choose a JPG or PNG image.";
  }
});