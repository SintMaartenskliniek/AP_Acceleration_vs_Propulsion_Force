"""
Function to calculate the trailing limb angle (TLA) for left and right sides.

Trailing limb angle is defined as the angle between the laboratory’s vertical axis 
and a vector created between the greater trochanter and the 5th metatarsal head (Hsiao et al. 2025).
Other commonly used proximal landmarks are the pelvis COM or hip joint center, and as distal landmarks 
the foot COM, COP, or lateral malleolus (Kenworthy et al. 2025).

This function exports two metrics of the TLA (Cohen et al. 2024):
    - aTLA:     value of TLA at the time of maxAGRF (peak propulsion force)
    - tcTLA:    value of TLA at terminal contact (toe-off)
    - maxTLA:   maximum value of TLA

Version - Author:
    2025: Lars van Rengs - l.vanrengs@maartenskliniek.nl
"""

# Import dependencies
import numpy as np
import matplotlib.pyplot as plt


def trailinglimbangle(markerdata, fs_markerdata = 100, proximal_landmark="pelvis_com", distal_landmark="toe", **kwargs):
    
    # Set defaults
    fs_markerdata = 100 # Sample frequency of the marker data
    debugplot = False
    
    # Optional inputs
    gaitevents = kwargs.get("gaitevents", None)
    gaitcharacteristics = kwargs.get("gaitcharacteristics", None)
    debugplot = kwargs.get("debugplot", False)
            
    trailinglimbangle = {}


    # ----------------------------------------------------------
    # Mapping for proximal landmarks
    # ----------------------------------------------------------
    def get_proximal(markerdata, side, option):
        if isinstance(option, str):
            if option == "pelvis_com":      # mean of ASI + PSI
                if side == 'L':
                    return (markerdata['LASI'] + markerdata['LPSI']) / 2
                else:
                    return (markerdata['RASI'] + markerdata['RPSI']) / 2
            
            elif option == "HJC":           # hip joint center
                return markerdata[f"{side}HJC"]
            
            elif option == "GrTroch":       # greater trochanter
                return markerdata[f"{side}GrTroch"]
            
            else:
                raise ValueError(f"Unknown proximal_landmark option: {option}")
        
        else:
            return option


    # ----------------------------------------------------------
    # Mapping for distal landmarks
    # ----------------------------------------------------------
    def get_distal(markerdata, side, option):
        if isinstance(option, str):
            if option == "toe":             # toe marker
                return markerdata[f"{side}TOE"]
            
            elif option == "MT2":           # 2nd metatarsal head
                return markerdata[f"{side}MT2"]
            
            elif option == "MT5":           # 5th metatarsal head
                return markerdata[f"{side}MT5"]
            
            elif option == "ankle":         # ankle marker
                return markerdata[f"{side}ANK"]

            elif option == "foot_com":      # mean of TOE + HEE
                if side == 'L':
                    return (markerdata['LTOE'] + markerdata['LHEE']) / 2
                else:
                    return (markerdata['RTOE'] + markerdata['RHEE']) / 2
            
            else:
                raise ValueError(f"Unknown distal_landmark option: {option}")
        
        else:
            return option
      

    # ----------------------------------------------------------
    # Helper function to compute TLA signal
    # ----------------------------------------------------------
    def compute_TLA(proximal, distal):
        vec = proximal - distal  # vector proximal → distal
        TLA = np.degrees(np.arctan2(-vec[:,1], vec[:,2]))
        return TLA


    # ----------------------------------------------------------
    # Compute TLA time series for left and right leg
    # ----------------------------------------------------------
    proximal_L = get_proximal(markerdata, 'L', proximal_landmark)
    proximal_R = get_proximal(markerdata, 'R', proximal_landmark)

    distal_L = get_distal(markerdata, 'L', distal_landmark)
    distal_R = get_distal(markerdata, 'R', distal_landmark)

    TLA_left  = compute_TLA(proximal_L, distal_L)
    TLA_right = compute_TLA(proximal_R, distal_R)

    trailinglimbangle['Trailing Limb Angle left (deg)'] = TLA_left
    trailinglimbangle['Trailing Limb Angle right (deg)'] = TLA_right


    # ----------------------------------------------------------
    # If gait events and characteristics are defined
    # ----------------------------------------------------------
    if gaitevents is not None and gaitcharacteristics is not None:
        # ----------------------------------------------------------
        # Compute aTLA for each peak anterior GRF
        # ----------------------------------------------------------
        idx_peak_left = gaitcharacteristics['Peak propulsion left'][:, 0].astype(int)
        idx_peak_right = gaitcharacteristics['Peak propulsion right'][:, 0].astype(int)
    
        aTLA_left_vals = np.array([TLA_left[idx] if idx < len(TLA_left) else np.nan for idx in idx_peak_left])
        aTLA_right_vals = np.array([TLA_right[idx] if idx < len(TLA_right) else np.nan for idx in idx_peak_right])
    
        aTLA_left = np.column_stack((idx_peak_left, aTLA_left_vals))
        aTLA_right = np.column_stack((idx_peak_right, aTLA_right_vals))
    
    
        # ----------------------------------------------------------
        # Compute tcTLA for each terminal contact (toe-off)
        # ----------------------------------------------------------
        idx_tc_left = np.sort(gaitevents["Index numbers terminal contact left"])
        idx_tc_right = np.sort(gaitevents["Index numbers terminal contact right"])
    
        tcTLA_left_vals = np.array([TLA_left[idx] if idx < len(TLA_left) else np.nan for idx in idx_tc_left])
        tcTLA_right_vals = np.array([TLA_right[idx] if idx < len(TLA_right) else np.nan for idx in idx_tc_right])
    
        tcTLA_left = np.column_stack((idx_tc_left, tcTLA_left_vals))
        tcTLA_right = np.column_stack((idx_tc_right, tcTLA_right_vals))
    
    
        # ----------------------------------------------------------
        # Compute maxTLA for each gait cycle
        # ----------------------------------------------------------
        IC_left = np.sort(gaitevents["Index numbers initial contact left"])
        IC_right = np.sort(gaitevents["Index numbers initial contact right"])
    
        # For each stride (between consecutive ICs)
        maxTLA_left = []
        for i in range(len(IC_left) - 1):
            start, end = IC_left[i], IC_left[i + 1]
            if end > len(TLA_left):  # safeguard
                end = len(TLA_left)
            segment = TLA_left[start:end]
            if np.all(np.isnan(segment)):
                maxTLA_left.append([np.nan, np.nan])
            else:
                local_idx = np.nanargmax(segment)
                global_idx = start + local_idx
                maxTLA_left.append([global_idx, segment[local_idx]])
        maxTLA_left = np.array(maxTLA_left)
    
        maxTLA_right = []
        for i in range(len(IC_right) - 1):
            start, end = IC_right[i], IC_right[i + 1]
            if end > len(TLA_right):
                end = len(TLA_right)
            segment = TLA_right[start:end]
            if np.all(np.isnan(segment)):
                maxTLA_right.append([np.nan, np.nan])
            else:
                local_idx = np.nanargmax(segment)
                global_idx = start + local_idx
                maxTLA_right.append([global_idx, segment[local_idx]])
        maxTLA_right = np.array(maxTLA_right)
    
    
        # ----------------------------------------------------------
        # Add to output dictionary
        # ----------------------------------------------------------
        trailinglimbangle['aTLA left (deg)'] = aTLA_left
        trailinglimbangle['aTLA right (deg)'] = aTLA_right
        trailinglimbangle['tcTLA left (deg)'] = tcTLA_left
        trailinglimbangle['tcTLA right (deg)'] = tcTLA_right
        trailinglimbangle['maxTLA left (deg)'] = maxTLA_left
        trailinglimbangle['maxTLA right (deg)'] = maxTLA_right
    
    
        # ----------------------------------------------------------
        # Optional Debug Plot
        # ----------------------------------------------------------
        if debugplot:
            time = np.arange(len(TLA_left)) / fs_markerdata
            fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
            # --- Left leg ---
            axs[0].plot(time, TLA_left, label='TLA Left')
            axs[0].plot(aTLA_left[:,0] / fs_markerdata, aTLA_left[:,1], 'r*', markersize=10, label='aTLA')
            axs[0].plot(tcTLA_left[:, 0] / fs_markerdata, tcTLA_left[:, 1], 'md', markersize=6, label='tcTLA')
            axs[0].plot(maxTLA_left[:,0] / fs_markerdata, maxTLA_left[:,1], 'bo', markersize=8, label='maxTLA')
            axs[0].set_ylabel('Angle (deg)')
            axs[0].set_title('Trailing Limb Angle - Left')
            axs[0].legend()
            axs[0].grid(True)
            axs[0].set_xlim(30, 40)
    
            # --- Add gait events for left leg ---
            IC_left_time = np.array(gaitevents["Index numbers initial contact left"]) / fs_markerdata
            TC_left_time = np.array(gaitevents["Index numbers terminal contact left"]) / fs_markerdata
    
            for t in IC_left_time:
                axs[0].axvline(t, color='k', linestyle='--', linewidth=1, alpha=0.7, label='IC Left' if t == IC_left_time[0] else "")
            for t in TC_left_time:
                axs[0].axvline(t, color='g', linestyle='--', linewidth=1, alpha=0.7, label='TC Left' if t == TC_left_time[0] else "")
    
            # --- Right leg ---
            axs[1].plot(time, TLA_right, label='TLA Right')
            axs[1].plot(aTLA_right[:,0] / fs_markerdata, aTLA_right[:,1], 'r*', markersize=10, label='aTLA')
            axs[1].plot(tcTLA_right[:, 0] / fs_markerdata, tcTLA_right[:, 1], 'md', markersize=6, label='tcTLA')
            axs[1].plot(maxTLA_right[:,0] / fs_markerdata, maxTLA_right[:,1], 'bo', markersize=8, label='maxTLA')
            axs[1].set_ylabel('Angle (deg)')
            axs[1].set_xlabel('Time (s)')
            axs[1].set_title('Trailing Limb Angle - Right')
            axs[1].legend()
            axs[1].grid(True)
            axs[1].set_xlim(30, 40)
    
            # --- Add gait events for right leg ---
            IC_right_time = np.array(gaitevents["Index numbers initial contact right"]) / fs_markerdata
            TC_right_time = np.array(gaitevents["Index numbers terminal contact right"]) / fs_markerdata
    
            for t in IC_right_time:
                axs[1].axvline(t, color='k', linestyle='--', linewidth=1, alpha=0.7, label='IC Right' if t == IC_right_time[0] else "")
            for t in TC_right_time:
                axs[1].axvline(t, color='g', linestyle='--', linewidth=1, alpha=0.7, label='TC Right' if t == TC_right_time[0] else "")
    
            plt.tight_layout()
            plt.show()

    
    return trailinglimbangle
        
