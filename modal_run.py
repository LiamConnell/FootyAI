# run_modal_train.py
import modal
from src.v2.train import main

# You can choose the GPU type: "A10G", "A100", "T4", etc.
gpu_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install("torch", "pydantic", "python-dotenv", "tqdm", "matplotlib", "pillow", "rich")
    .add_local_dir("./src", remote_path="/src")
)

# gpu_image = gpu_image.add_local_dir("./src", remote_path="/src")
# print(gpu_image.workdir("/"))

app = modal.App(name="torch-train")

@app.function(
    image=gpu_image,
    gpu="A10G",           # or "A100", "T4" 
    timeout=60 * 60 * 4,  # 4 hour timeout for longer training
    # mounts=[modal.Mount.from_local_dir("src", remote_path="/src")],
)
def train():
    main()

if __name__ == "__main__":
    app.run(train)
