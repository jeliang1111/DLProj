import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. LOAD AND PREP DATA FOR VAE (MinMax Scaling)
# ==========================================
feature_cols = ['Discharge Time (s)', 'Decrement 3.6-3.4V (s)', 'Max. Voltage Dischar. (V)', 
                'Min. Voltage Charg. (V)', 'Time at 4.15V (s)', 'Time constant current (s)', 'Charging time (s)']

# Load the separate datasets you uploaded
hnei_df = pd.read_csv('Battery_RUL_Cleaned.csv')
nasa_df = pd.read_csv('nasa_battery_cycles.csv')

# We need separate scalers for each to map their unique ranges to [0, 1]
scaler_hnei = MinMaxScaler(feature_range=(0, 1))
scaler_nasa = MinMaxScaler(feature_range=(0, 1))

hnei_scaled = scaler_hnei.fit_transform(hnei_df[feature_cols])
nasa_scaled = scaler_nasa.fit_transform(nasa_df[feature_cols])

# Add the scaled features back to copies of the dataframes so we can group by battery
hnei_df_scaled = hnei_df.copy()
nasa_df_scaled = nasa_df.copy()
hnei_df_scaled[feature_cols] = hnei_scaled
nasa_df_scaled[feature_cols] = nasa_scaled

# ==========================================
# 2. SEQUENCE GENERATOR (No Target/RUL Needed)
# ==========================================
def create_unsupervised_sequences(data_df, seq_length=10):
    """Creates (batch_size, 10, 7) arrays for VAE training."""
    X_seq = []
    
    # Assuming Battery_ID exists or was reconstructed like in previous scripts
    # If not, you can reconstruct it using: (data_df['Cycle_Index'] < data_df['Cycle_Index'].shift(1)).cumsum()
    if 'Battery_ID' not in data_df.columns:
        data_df['Battery_ID'] = (data_df['Cycle_Index'] < data_df['Cycle_Index'].shift(1)).cumsum()

    for bat_id, group in data_df.groupby('Battery_ID'):
        dynamic_features = group[feature_cols].values
        for i in range(len(group) - seq_length):
            X_seq.append(dynamic_features[i : i + seq_length])
            
    return np.array(X_seq)

seq_length = 10
X_hnei = create_unsupervised_sequences(hnei_df_scaled, seq_length)
X_nasa = create_unsupervised_sequences(nasa_df_scaled, seq_length)

print(f"HNEI Training Sequences: {X_hnei.shape}")
print(f"NASA Training Sequences: {X_nasa.shape}")

# ==========================================
# 3. DEFINE THE SEQ2SEQ VAE ARCHITECTURE
# ==========================================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_seq_vae(seq_length=10, num_features=7, latent_dim=4):
    # Encoder
    encoder_inputs = layers.Input(shape=(seq_length, num_features))
    x = layers.LSTM(32, activation='tanh', return_sequences=False)(encoder_inputs)
    x = layers.Dense(16, activation='relu')(x)
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling(name="z")([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.RepeatVector(seq_length)(latent_inputs)
    x = layers.LSTM(32, activation='tanh', return_sequences=True)(x)
    
    # Sigmoid guarantees the output is strictly bounded between [0, 1]
    decoder_outputs = layers.TimeDistributed(layers.Dense(num_features, activation='sigmoid'))(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    # VAE Custom Training Loop
    class VAE(Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.mse(data, reconstruction), axis=1))
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = reconstruction_loss + kl_loss
                
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            return {"loss": self.total_loss_tracker.result()}

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return vae, decoder

# ==========================================
# 4. BUILD AND TRAIN THE TWO VAES
# ==========================================
print("\nTraining HNEI VAE...")
vae_hnei, decoder_hnei = build_seq_vae()
vae_hnei.fit(X_hnei, epochs=30, batch_size=64, verbose=1)

print("\nTraining NASA VAE...")
vae_nasa, decoder_nasa = build_seq_vae()
vae_nasa.fit(X_nasa, epochs=30, batch_size=64, verbose=1)

# ==========================================
# 5. SIMULATION / DEMO GENERATOR
# ==========================================
def generate_synthetic_battery(decoder, scaler, dataset_name):
    """Samples random latent noise, decodes it into a valid [0, 1] trajectory, 
       and reverses the scaling so you can view the physical values."""
    
    # 1. Sample pure random noise (1 sample, 4 latent dimensions)
    random_latent_vector = np.random.normal(size=(1, 4))
    
    # 2. Decode the noise into a 10x7 normalized trajectory
    synthetic_normalized_trajectory = decoder.predict(random_latent_vector, verbose=0)
    
    # 3. Inverse transform it back to real physical units (seconds, volts) for human viewing
    # Note: synthetic_normalized_trajectory is shape (1, 10, 7). We extract the (10, 7) array for the scaler.
    synthetic_physical_trajectory = scaler.inverse_transform(synthetic_normalized_trajectory[0])
    
    print(f"\n--- Generated 10-Cycle {dataset_name} Battery Trajectory ---")
    df = pd.DataFrame(synthetic_physical_trajectory, columns=feature_cols)
    print(df.head(5).to_string()) # Print first 3 cycles to verify
    
    return synthetic_normalized_trajectory

# Generate and view data for your demo!
fake_hnei_input = generate_synthetic_battery(decoder_hnei, scaler_hnei, "HNEI")
fake_nasa_input = generate_synthetic_battery(decoder_nasa, scaler_nasa, "NASA")