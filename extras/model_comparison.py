import matplotlib.pyplot as plt

# Models and their accuracy values (out of 100 scale)
models = [
    'SignSei (Proposed)',
    'ConvNet + LSTM Hybrid',
    'MediaPipe Hands Baseline',
    'YOLO-based Gesture Recognition',
    'DeepHand (RGB-D based)',
    'Skeleton-based (OpenPose)',
    'Sign Language Transformer (SLT)'
]

accuracies = [
    92.3,  # SignSei (Proposed)
    85.4,  # ConvNet + LSTM Hybrid
    70.3,  # MediaPipe Hands Baseline
    75.5,  # YOLO
    80.2,  # DeepHand
    78.2,  # Skeleton-based
    89.0   # SLT
]

# Bar colors - highlight your model
colors = ['green' if model == 'SignSei (Proposed)' else 'gray' for model in models]

plt.figure(figsize=(12, 6))
bars = plt.barh(models, accuracies, color=colors)

# Annotate each bar with its accuracy value (formatted to 1 decimal place)
for bar in bars:
    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.1f}%', va='center', fontsize=9)

plt.xlabel('Accuracy (%)')
plt.title('Model Comparison: SignSei vs Existing Models')
plt.xlim(0, 100)  # Set limit from 0 to 100
plt.gca().invert_yaxis()  # Highest at the top

plt.tight_layout()
plt.savefig('model_comparison_out_of_100.png', dpi=300)
plt.show()
