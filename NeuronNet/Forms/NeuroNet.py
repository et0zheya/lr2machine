import tkinter as tk
from tkinter import messagebox, Toplevel

from Service import train, validation
from .Metric import MetricWindow


class NeuralNetworkWindow:
    def __init__(self, master):
        self.master = Toplevel(master)
        self.master.title("Neural Network")
        self.master.geometry("500x400")
        self.master.grab_set()

        # Create frames for different areas with borders
        self.epochs_frame = tk.LabelFrame(self.master, text="Epochs Input", padx=10, pady=10)
        self.epochs_frame.place(relx=0.5, rely=0.1, anchor='n')

        self.learning_frame = tk.LabelFrame(self.master, text="Learning Options", padx=10, pady=10)
        self.learning_frame.place(relx=0.5, rely=0.4, anchor='n')

        self.validation_frame = tk.LabelFrame(self.master, text="Validation", padx=10, pady=10)
        self.validation_frame.place(relx=0.3, rely=0.7, anchor='n')

        self.metrics_frame = tk.LabelFrame(self.master, text="Metrics", padx=10, pady=10)
        self.metrics_frame.place(relx=0.7, rely=0.7, anchor='n')

        # Epochs input
        tk.Label(self.epochs_frame, text="Epochs:").pack(side='left')
        self.epochs_entry = tk.Entry(self.epochs_frame)
        self.epochs_entry.pack(side='left')

        # Learning rate input
        tk.Label(self.epochs_frame, text="Learning Rate:").pack(side='left')
        self.speed_entry = tk.Entry(self.epochs_frame)
        self.speed_entry.pack(side='bottom')

        # Buttons for learning processes
        learn_new_button = tk.Button(self.learning_frame, text="Train from Scratch", command=self.train_from_scratch)
        learn_new_button.pack(side='left', padx=5)

        learn_button = tk.Button(self.learning_frame, text="Retrain", command=self.retrain)
        learn_button.pack(side='left', padx=5)

        # Validation button
        validate_button = tk.Button(self.validation_frame, text="Validate", command=self.validate_model)
        validate_button.pack(padx=5, pady=5)  # Center the button within the frame

        # Graphs button in metrics frame
        graphs_button = tk.Button(self.metrics_frame, text="Training Graphs", command=self.show_graphs_for_train)
        graphs_button.pack(padx=5, pady=5)  # Center the button within the frame

        graphs_button = tk.Button(self.metrics_frame, text="Validation Graphs", command=self.show_graphs_for_validation)
        graphs_button.pack(padx=5, pady=5)  # Center the button within the frame

    def train_from_scratch(self):
        if self.confirm_action("Are you sure you want to start training from scratch?"):
            epochs = self.get_epochs()
            speed = self.get_speed()
            if epochs is not None:
                print(f"Train from scratch for {epochs} epochs.")
                # Add your training logic here
                train(speed, epochs, True)

    def retrain(self):
        if self.confirm_action("Are you sure you want to start training without zeroing weights?"):
            epochs = self.get_epochs()
            speed = self.get_speed()
            if epochs and speed is not None:
                print(f"Training for {epochs} epochs without zeroing weights.")
                # Add your retraining logic here
                train(speed, epochs, False)

    def validate_model(self):
        # Validation logic can go here
        print("Validation process started...")
        epochs = self.get_epochs()
        validation(epochs)

    def get_speed(self):
        try:
            speed = float(self.speed_entry.get())
            return speed
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a valid number of speed.")
            return None

    def get_epochs(self):
        try:
            epochs = int(self.epochs_entry.get())
            return epochs
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a valid number of epochs.")
            return None

    @staticmethod
    def confirm_action(message):
        return messagebox.askyesno("Confirm Action", message)

    def show_graphs_for_train(self):
        MetricWindow(self.master, "train_metrics")

    def show_graphs_for_validation(self):
        MetricWindow(self.master, "validate_metrics")
