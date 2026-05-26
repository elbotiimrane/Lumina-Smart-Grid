with open("prediction/app.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace the corrupted headers
text = text.replace("YO? Multi-layered Network Topology", "🌐 Multi-layered Network Topology")
text = text.replace("Y\"S Model Comparison & Benchmarks", "📊 Model Comparison & Benchmarks")
text = text.replace("Y\"^ Time-Series Forecast Explorer", "📈 Time-Series Forecast Explorer")
text = text.replace("Y' Adaptive Dimming & Optimization", "💡 Adaptive Dimming & Optimization")
text = text.replace("YO? Interactive Graph Network Topologies", "🌐 Interactive Graph Network Topologies")
text = text.replace("Y\"S Spatio-Temporal Prediction Model Benchmarking", "📊 Spatio-Temporal Prediction Model Benchmarking")
text = text.replace("Y\"^ Interactive Time-Series Forecast Explorer", "📈 Interactive Time-Series Forecast Explorer")
text = text.replace("s Adaptive Lighting Dimming", "💡 Adaptive Lighting Dimming")
text = text.replace("Y\"< Academic Experiment", "📋 Academic Experiment")

with open("prediction/app.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Cleanup successfully completed!")
