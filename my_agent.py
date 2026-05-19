def agent(obs, config):
    # Inspect what's available on first call
    print("OBS keys:", list(obs.keys()) if hasattr(obs, 'keys') else dir(obs))
    print("CONFIG:", config)
    
    # Safe fallback — return 0 until you understand the action space
    return 0