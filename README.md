# AmbientMNIST

Run MNIST digit classification entirely on-device under ambient energy on an MSP430FR5994.

---

## ⚡ Demo
<!--- 동작하는 영상이랑 커맨드창에 제대로 분류되는지 프린트 되는 이미지 or 영상 -->

- **Solar-Harvested Inference**  
  MNIST test images classified in real time while powered only by a small solar panel.  

- **FRAM Checkpointing**  
  Robust progress across power failures via SONIC’s sparse undo-logging and FRAM-based task commits.

---

## 🎯 Motivation

Energy-harvesting IoT devices eliminate battery replacements, but unreliable power makes continuous computation difficult. We asked:

> How can you run a deep neural network end-to-end on a MCU powered only by ambient energy?

By combining CMU’s SONIC intermittent-compute framework with on-device MNIST inference, we demonstrate robust, accurate classification under varying energy conditions.

---

## 📖 Project Description

1. **Model Training & Header Generation**  
   - Train a small CNN on MNIST in TensorFlow  
   - Use `AmbientMNIST_script.ipynb` to quantize weights and write C header files

2. **SONIC Integration**  
   - Copy headers into the SONIC repo under `apps/my_mnist/params/`  
   - Compile SONIC example to produce `my_mnist.out`  

3. **On-Device Deployment**  
   - Flash `my_mnist.out` to the MSP430FR5994 via TI UniFlash  
   - Validate classifications under ambient solar harvesting

---

## 🔧 Hardware & Sensors

| Component                    | Part Number             | Role                                      |
|------------------------------|-------------------------|-------------------------------------------|
| Microcontroller              | MSP430FR5994            | Low-power MCU with FRAM checkpointing     |
| Energy Harvesting            | 50 mm² solar panel      | Ambient power source                      |

---

