const tabButtons = document.querySelectorAll(".tab-btn");
const panels = {
  image: document.querySelector('[data-tab-panel="image"]'),
  video: document.querySelector('[data-tab-panel="video"]'),
};

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const tab = btn.dataset.tab;
    Object.entries(panels).forEach(([name, panel]) => {
      panel.classList.toggle("hidden", name !== tab);
    });
  });
});

function setupDropzone(dropzoneId, inputId, filenameId) {
  const dropzone = document.getElementById(dropzoneId);
  const input = document.getElementById(inputId);
  const filenameLabel = document.getElementById(filenameId);

  input.addEventListener("change", () => {
    filenameLabel.textContent = input.files[0] ? input.files[0].name : "";
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );

  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
      input.files = e.dataTransfer.files;
      filenameLabel.textContent = file.name;
    }
  });
}

setupDropzone("image-dropzone", "image-input", "image-filename");
setupDropzone("video-dropzone", "video-input", "video-filename");

function bindRangeDisplay(rangeId, displayId, decimals) {
  const range = document.getElementById(rangeId);
  const display = document.getElementById(displayId);
  range.addEventListener("input", () => {
    display.textContent = decimals === 0 ? range.value : Number(range.value).toFixed(decimals);
  });
}

bindRangeDisplay("image-confidence", "image-confidence-value", 2);
bindRangeDisplay("video-confidence", "video-confidence-value", 2);
bindRangeDisplay("video-frame-skip", "video-frame-skip-value", 0);

const statusMessage = document.getElementById("status-message");
const loading = document.getElementById("loading");
const loadingText = document.getElementById("loading-text");
const resultContainer = document.getElementById("result-container");

function setStatus(message, isError = false) {
  statusMessage.textContent = message;
  statusMessage.classList.toggle("error", isError);
}

function setLoading(isLoading, text) {
  loading.classList.toggle("hidden", !isLoading);
  if (text) loadingText.textContent = text;
}

function updateCounts(counts) {
  document.getElementById("count-clean").textContent = counts["unoccupied-clean"] ?? 0;
  document.getElementById("count-occupied").textContent = counts["occupied"] ?? 0;
  document.getElementById("count-dirty").textContent = counts["unoccupied-dirty"] ?? 0;
}

async function submitForm(endpoint, formData, { loadingText: text, onResult, submitBtn }) {
  setStatus("");
  resultContainer.innerHTML = "";
  setLoading(true, text);
  submitBtn.disabled = true;

  try {
    const response = await fetch(endpoint, { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Detection failed.");
    }

    onResult(data);
    updateCounts(data.counts);
    setStatus("Detection complete.");
  } catch (err) {
    setStatus(err.message || "Something went wrong.", true);
  } finally {
    setLoading(false);
    submitBtn.disabled = false;
  }
}

document.getElementById("image-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("image-input");
  if (!fileInput.files[0]) {
    setStatus("Please choose an image first.", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("confidence", document.getElementById("image-confidence").value);
  formData.append("draw_mode", document.getElementById("image-draw-mode").value);

  submitForm("/api/detect/image", formData, {
    loadingText: "Running detection on image…",
    submitBtn: document.getElementById("image-submit"),
    onResult: (data) => {
      const img = document.createElement("img");
      img.src = `data:image/jpeg;base64,${data.image_base64}`;
      resultContainer.appendChild(img);
    },
  });
});

document.getElementById("video-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("video-input");
  if (!fileInput.files[0]) {
    setStatus("Please choose a video first.", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("confidence", document.getElementById("video-confidence").value);
  formData.append("draw_mode", document.getElementById("video-draw-mode").value);
  formData.append("frame_skip", document.getElementById("video-frame-skip").value);

  submitForm("/api/detect/video", formData, {
    loadingText: "Processing video — this can take a while…",
    submitBtn: document.getElementById("video-submit"),
    onResult: (data) => {
      const video = document.createElement("video");
      video.src = data.video_url;
      video.controls = true;
      video.autoplay = true;
      resultContainer.appendChild(video);

      const timeline = data.counts_timeline || [];
      const fps = data.fps || 25;
      if (timeline.length > 0) {
        video.addEventListener("timeupdate", () => {
          const frameIndex = Math.min(
            Math.floor(video.currentTime * fps),
            timeline.length - 1
          );
          updateCounts(timeline[Math.max(frameIndex, 0)]);
        });
      }
    },
  });
});
