from a_imports import *

# Get all the data from manifest
rows = read_manifest(g_configs.MANIFEST)
train_rows, val_rows, test_rows = split(rows, g_configs.TRAIN_FRAC, g_configs.VAL_FRAC)

# Make dataset and data loaders
train_dataset = dataset(train_rows, g_configs.DATA_DIR, g_configs.RESOLUTION, g_configs.BLOB_SIGMA, augment = True)
val_dataset = dataset(val_rows, g_configs.DATA_DIR, g_configs.RESOLUTION, g_configs.BLOB_SIGMA, augment = False)

train_loader = DataLoader(train_dataset, batch_size = g_configs.BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = g_configs.BATCH_SIZE, shuffle = False)

# Load CNN
model = CNN().to(g_configs.DEVICE)

# Using Smooth L1 Loss criterion 
criterion = nn.SmoothL1Loss()

# Using Adam Optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=g_configs.LR, weight_decay=g_configs.WEIGHT_DECAY)

# Make the checkpoint dir
os.makedirs(g_configs.CKPT_DIR, exist_ok = True)

# Training Loop
'''
We follow the standard training loop using Adam optimizer and Smooth L1 Loss
1. Call the model
2. Initialize the total loss (that has to be minimized) at the current iteration
3. Load images and targets to the GPU 
4. Standard 5 lines for every training loop:
        predictions = model(images)
        loss = criterion(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
5. Running Loss Arithmetic:
    Our aim is to compute the average mean loss per epoch
    So for that we simply compute the weighted loss, i.e loss for one item times number of items
    and then divide it by total number of items in the entire dataset

    So, for each batch, running_loss(current_batch) = running_loss(prev_batch)+ loss.item()*images.size() {images is the 
    array of items in a batch}

    Our aim is to compare the total avg loss per epoch, it should be lesser than the previous one each time
'''


best_validation_loss = float("inf")
for epochs in range(g_configs.EPOCHS):
    # Training Loop
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images = images.to(g_configs.DEVICE)
        targets = targets.to(g_configs.DEVICE)

        predictions = model(images)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)
    running_loss /= len(train_dataset)

    # Validation Loop
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(g_configs.DEVICE)
            targets = targets.to(g_configs.DEVICE)

            predictions = model(images)
            validation_loss += criterion(predictions, targets).item() * images.size(0)
    validation_loss /= len(val_dataset)

    '''
    Now we need the output of the main thing we need to diagnose in every iteration:
    1. Epoch Number
    2. Training Loss Value
    3. Validation Loss Value

    Now if the Validation loss value is lesser than the Lowest (yet) Validation loss, it becomes
    the new lowest and we save the kernel values and all in the checkpoint (config.CKPT_PATH) directory
    '''

    # I initialized the best validation loss right before the loop as + infinity so that first epoch's val loss becomes the best at first 
    # and then the next best becomes the best, ykwim?
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        torch.save(model.state_dict(), g_configs.CKPT_PATH)

    # I decided to put the print part in the end so that we can also get the updated best val loss
    print(f"EPOCH: [{epochs+1}/{g_configs.EPOCHS}]\nTRAIN LOSS: {running_loss:.4f}\nVALIDATION LOSS: {validation_loss:.4f}\nLeast Validation Loss Yet: {best_validation_loss:.4f}")







