

device = ("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 100
learning_rate = 0.00001
batch_size = 256
shuffle = True
pin_memory = True
num_workers = 1

transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
        ]
    )

dataset = MSADataset(
    root_dir="datasets/humanchr1430covMSA_center_base_dataset_w51_h100_n64000_not_human_readable",
    annotation_file="datasets/humanchr1430covMSA_center_base_dataset_w51_h100_n64000_not_human_readable/train_labels.csv",
    transform=transform
    )

train_set, validation_set = torch.utils.data.random_split(dataset,[52000,12000])
train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory)
validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle, batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)

# Model
model = Net()
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()
print("Starting the training process.")

# Train Network
for epoch in range(num_epochs):
    losses = []
    
    epoch_start_time = time.time()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        losses.append(loss.item())
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent or adam step
        optimizer.step()
    
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f'Cost at epoch {epoch} is {sum(losses)/len(losses)}. Training the epoch took {epoch_duration} seconds.')

    if (epoch+1) % 10 == 0:
        print(f"Accuracy on the validation set after epoch {epoch}:") 
        check_accuracy(validation_loader, model)

end_time = time.time()
duration = end_time - start_time

print(f"Finished training. The process took {duration} seconds.")    

print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(validation_loader, model)

save_path = "trained_models"
if not os.path.exists(save_path):
    os.makedirs(save_path)

torch.save(model.state_dict(), os.path.join(save_path, "simple_conv_net_humanchr1430_center_base_w51_h100_n64000_not_human_readable"))
