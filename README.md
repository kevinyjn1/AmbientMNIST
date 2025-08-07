# AmbientMNIST

Run MNIST digit classification entirely on-device under ambient energy, using CMU’s SONIC framework on an MSP430FR5994.

---

## ⚡ Demo

- **Solar-Harvested Inference**  
  MNIST test images classified in real time while powered only by a small solar panel.  
- **Interrupt-Driven Wakeup**  
  A thermal TAS sensor triggers inference only when motion or heat is detected, slashing idle power.  
- **FRAM Checkpointing**  
  Robust progress across power failures via SONIC’s sparse undo-logging and FRAM-based task commits.

---

## 🎯 Motivation

Energy-harvesting IoT devices eliminate battery replacements, but unreliable power makes continuous computation difficult. We asked:

> How can you run a deep neural network end-to-end on a sub-threshold MCU powered only by ambient energy?

By combining CMU’s SONIC intermittent-compute framework with on-device MNIST inference, we demonstrate robust, accurate classification under varying energy conditions.

---

## 📖 Project Description

1. **Model Training & Header Generation**  
   - Train a small CNN on MNIST in TensorFlow  
   - Use `genheader.py` to quantize weights and write C header files

2. **SONIC Integration**  
   - Copy headers into the SONIC repo under `examples/my_mnist/params/`  
   - Compile SONIC example to produce `my_mnist.out`  

3. **On-Device Deployment**  
   - Flash `my_mnist.out` to the MSP430FR5994 via TI UniFlash  
   - Validate classifications under ambient solar harvesting

4. **Event-Driven Sensing**  
   - Hook up a Grid-EYE TAS thermal array over I²C  
   - Use SONIC’s interrupt service routine to wake and infer only on heat/motion

---

## 🔧 Hardware & Sensors

| Component                    | Part Number             | Role                                      |
|------------------------------|-------------------------|-------------------------------------------|
| Microcontroller              | MSP430FR5994            | Low-power MCU with FRAM checkpointing     |
| Energy Harvesting            | 50 mm² solar panel      | Ambient power source                      |
| Thermal Array Sensor (TAS)   | Panasonic Grid-EYE      | Event trigger (heat/motion)               |
| Power Management             | Buck/boost harvester    | Regulate and store ambient energy         |

---

## 📂 Repository Structure

