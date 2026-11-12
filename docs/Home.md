CUTracer is a debugging toolkit for NVIDIA GPU kernels, built on NVBit. Its runtime library instruments CUDA kernels at the SASS level without rebuilding the application. The `cutracer` command records per-warp traces, exposes data races with delay injection (`stress`, `reduce`), stops hung kernels with a per-warp report, runs NVIDIA Compute Sanitizer, and diagnoses a failing test end to end (`diagnose`).

🎯 Get started:

- [Quickstart](Quickstart.md)
- [Concepts](Concepts.md)
- [Examples](Examples.md)

📚 Reference:

- [Configuration](Configuration.md)
- [Instrumentation Modes](Instrumentation-Modes.md)
- [Instrumentation Plans](Instrumentation-Plans.md)
- [Outputs and File Formats](Outputs-and-File-Formats.md)
- [API and Data Structures](API-and-Data-Structures.md)

🔬 Analyses and workflows:

- [Analyses](Analyses.md)
- [Automated Diagnosis Service](Automated-Diagnosis-Service.md)
- [Post-processing: IPC Merge](Post-processing-IPC-Merge.md)
- [Triton_Proton Integration](Triton_Proton-Integration.md)

🔧 Architecture and development:

- [Architecture](Architecture.md)
- [Developer Guide](Developer-Guide.md)
- [Build, Test, and CI](Build-Test-and-CI.md)
- [Code Formatting](Code-Formatting.md)

❓ Help:

- [Troubleshooting](Troubleshooting.md)
- [FAQ](FAQ.md)

If you only have 5 minutes, read Quickstart and Analyses.

Welcome to the CUTracer wiki!
