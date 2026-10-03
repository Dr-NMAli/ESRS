"""Central configuration. Every hyperparameter in one place."""

SEED          = 42
N_ITEMS       = 5000
N_USERS       = 1000
CAT_SIZES     = {"Electronics": 2200, "Home": 1600, "Clothing": 1200}

# interaction simulation
ZIPF_S        = 1.2
N_INTER_MIN   = 20
N_INTER_MAX   = 120
N_INTER_MEAN  = 43.7

# encoder
D_EMB         = 128
TF_DIM        = 128
AE_HIDDEN     = 256
AE_LR         = 1e-3
AE_BATCH      = 64
AE_EPOCHS     = 200
AE_PATIENCE   = 20
AE_VAL_SPLIT  = 0.10

# ranking / evaluation 
ALPHA         = 0.6
K             = 10
TAU           = 0.7
ECO_WEIGHTS   = (0.5, 0.3, 0.2)     # (carbon, recycled, energy)
N_CANDIDATES  = 100
N_POS_PER_USER = 11

# explanation thresholds
THETA_ENV     = 0.70
THETA_SAV     = 10.0
THETA_POP     = 0.65