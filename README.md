# AmbientMNIST

Run MNIST digit classification entirely on-device under ambient energy on an MSP430FR5994.

---

## ⚡ Demo
<!--- 동작하는 영상이랑 커맨드창에 제대로 분류되는지 프린트 되는 이미지 or 영상 -->

![AmbientMNIST in action](./ambientMNIST/demo.gif)

- **Solar-Harvested Inference**  
  MNIST test images classified in real time while powered only by a small solar panel.  

- **FRAM Checkpointing**  
  Robust progress across power failures via SONIC’s sparse undo-logging and FRAM-based task commits.

---

## 🎯 Motivation

The growing demand for intelligent IoT devices calls for ultra-low-power systems that can operate without batteries. Ambient energy harvesting, which draws power from sources like light or RF signals, offers a sustainable solution—eliminating the need for battery replacement and enabling long-term deployment in remote areas.

Yet, deploying neural networks on such systems is challenging due to limited memory and intermittent power. The MSP430FR5994, with its non-volatile FRAM and low-power design, provides a suitable platform to explore these constraints.

As a proof-of-concept, we implement a lightweight MNIST classifier to demonstrate that neural inference is possible under intermittent, harvested power. While MNIST is a simple task, it effectively validates the feasibility of energy-autonomous AI at the extreme edge.

---

## 📖 Project Description
To execute the model under intermittent power conditions, we implemented the inference logic using the SONIC framework. SONIC enables reliable DNN execution on energy-harvesting devices by maintaining two alternating buffers for intermediate feature maps, allowing progress to continue seamlessly across power cycles.

We integrated the layer-by-layer inference logic into main.c, ensuring that each operation fits within the available energy budget and memory. By leveraging SONIC’s double-buffering and task-based checkpointing mechanism, the system can recover from power loss at fine-grained computation boundaries without restarting the entire inference process.

The overall workflow is as follows:

1. **Lightweight CNN & Header Generation**  
   - Train a 2-conv, 2-FC CNN on MNIST in TensorFlow, then quantize to 5-bit fixed-point.  
   - Run `AmbientMNIST_script.ipynb` to serialize each weight/bias array into C header files (`*.h`).

2. **SONIC Framework Integration**  
   - Place headers under `apps/my_mnist/params/` in the SONIC repo.  
   - In `main.c`, implement each DNN layer as a SONIC task, using the framework’s dual-buffer scheme:  
     - **Buffer A/B** alternate reading/writing of intermediate feature maps.  
     - After each small “task,” SONIC checkpoints state to FRAM via sparse undo-logging.

3. **Compile & Deploy**  
   - Build with `make my_mnist` to generate `my_mnist.out`.  
   - Flash to the MSP430FR5994 using TI UniFlash.

4. **Robust Inference under Intermittent Power**  
   - A tiny solar panel charges a storage capacitor; when voltage dips, SONIC seamlessly retries the next task on recharge.  
   - At the end of inference, results are printed over UART for validation.

---

## 🔧 Hardware & Sensors

| Component                    | Part Number             | Role                                      |
|------------------------------|-------------------------|-------------------------------------------|
| Microcontroller              | MSP430FR5994            | Low-power MCU with FRAM checkpointing     |
| Energy Harvesting            | 50 mm² solar panel      | Ambient power source                      |

---

