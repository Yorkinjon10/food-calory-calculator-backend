# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation accuracy: {val_acc*100:.2f}%")
