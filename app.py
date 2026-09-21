import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
import os
import warnings

# Scikit-learn সতর্কবার্তা বন্ধ রাখা
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 1. Page Configuration
st.set_page_config(page_title="T20 Simulator & Match Archive", page_icon="🏏", layout="wide")
st.title("🏏 T20 Cricket Match Simulation Engine")

CSV_LOG_FILE = "simulated_matches_ball_by_ball.csv"

# 2. Load Model Bundle
@st.cache_resource
def load_bundle():
    return joblib.load('t20_simulator_best_model.joblib')

bundle = load_bundle()
model = bundle['model']
encoder = bundle['encoder']
reverse_map = bundle['reverse_outcome_map']
feature_names = bundle.get('features', [
    'innings', 'over', 'ball_in_over', 'legal_ball_number', 
    'current_score', 'wickets_lost', 'current_run_rate', 
    'batting_team_enc', 'bowling_team_enc'
])

# 3. 16 Elite Teams Playing XIs (Historical Rosters)
OFFICIAL_PLAYING_XIS = {
    "(M) Australia": ['DA Warner', 'AJ Finch', 'MR Marsh', 'MS Wade', 'SPD Smith', 'AC Agar', 'SR Watson', 'A Zampa', 'MA Starc', 'JR Hazlewood', 'PJ Cummins'],
    "(M) England": ['JC Buttler', 'EJG Morgan', 'AD Hales', 'DJ Malan', 'JM Bairstow', 'CJ Jordan', 'SM Curran', 'AU Rashid', 'SCJ Broad', 'JC Archer', 'GP Swann'],
    "(M) India": ['V Kohli', 'RG Sharma', 'SA Yadav', 'KL Rahul', 'MS Dhoni', 'HH Pandya', 'AR Patel', 'Arshdeep Singh', 'JJ Bumrah', 'YS Chahal', 'Kuldeep Yadav'],
    "(M) New Zealand": ['MJ Guptill', 'KS Williamson', 'TL Seifert', 'LRPL Taylor', 'GD Phillips', 'TG Southee', 'IS Sodhi', 'MJ Santner', 'TA Boult', 'JA Duffy', 'LH Ferguson'],
    "(M) Pakistan": ['Babar Azam', 'Mohammad Rizwan', 'Shoaib Malik', 'Fakhar Zaman', 'Umar Akmal', 'Shaheen Shah Afridi', 'Shadab Khan', 'Haris Rauf', 'Saeed Ajmal', 'Shahid Afridi', 'Umar Gul'],
    "(M) South Africa": ['Q de Kock', 'RR Hendricks', 'DA Miller', 'JP Duminy', 'AK Markram', 'D Pretorius', 'GF Linde', 'L Ngidi', 'T Shamsi', 'K Rabada', 'DW Steyn'],
    "(M) Sri Lanka": ['P Nissanka', 'MDKJ Perera', 'TM Dilshan', 'BKG Mendis', 'KC Sangakkara', 'PWH de Silva', 'NLTC Perera', 'SL Malinga', 'PVD Chameera', 'KMDN Kulasekara', 'BAW Mendis'],
    "(M) West Indies": ['N Pooran', 'R Powell', 'CH Gayle', 'BA King', 'SO Hetmyer', 'JO Holder', 'DJ Bravo', 'AJ Hosein', 'R Shepherd', 'AS Joseph', 'AD Russell'],
    "(W) Australia": ['BL Mooney', 'AJ Healy', 'MM Lanning', 'TM McGrath', 'EJ Villani', 'EA Perry', 'A Gardner', 'ML Schutt', 'JL Jonassen', 'G Wareham', 'S Molineux'],
    "(W) England": ['HC Knight', 'TT Beaumont', 'AE Jones', 'SIR Dunkley', 'SJ Taylor', 'S Ecclestone', 'KH Brunt', 'S Glenn', 'A Shrubsole', 'CE Dean', 'NR Sciver'],
    "(W) India": ['S Mandhana', 'H Kaur', 'JI Rodrigues', 'Shafali Verma', 'RM Ghosh', 'DB Sharma', 'P Vastrakar', 'RP Yadav', 'Poonam Yadav', 'Renuka Singh', 'N Shree Charani'],
    "(W) New Zealand": ['ML Green', 'AE Satterthwaite', 'KJ Martin', 'GE Plimmer', 'BM Halliday', 'AC Kerr', 'SFM Devine', 'LMM Tahuhu', 'LM Kasperek', 'HNK Jensen', 'JM Kerr'],
    "(W) Pakistan": ['Bismah Maroof', 'Aliya Riaz', 'Muneeba Ali', 'Javeria Khan', 'Sidra Ameen', 'Nida Dar', 'Fatima Sana', 'Sadia Iqbal', 'Nashra Sandhu', 'Anam Amin', 'Sana Mir'],
    "(W) South Africa": ['L Wolvaardt', 'T Brits', 'L Lee', 'M du Preez', 'A Bosch', 'M Kapp', 'N de Klerk', 'S Ismail', 'N Mlaba', 'A Khaka', 'D van Niekerk'],
    "(W) Sri Lanka": ['H Madavi', 'NND de Silva', 'GWHM Perera', 'RMVD Gunaratne', 'MAA Sanjeewani', 'WK Dilhari', 'AC Jayangani', 'I Ranaweera', 'KDU Prabodhani', 'BMSM Kumari', 'HASD Siriwardene'],
    "(W) West Indies": ['SA Campbelle', 'CN Nation', 'Kycia A Knight', 'Q Joseph', 'B Cooper', 'HK Matthews', 'ASS Fletcher', 'A Mohammed', 'DJS Dottin', 'SR Taylor', 'SS Connell']
}

teams = sorted(list(OFFICIAL_PLAYING_XIS.keys()))

# 4. Next Match ID Generator
def get_next_match_id():
    if os.path.exists(CSV_LOG_FILE):
        try:
            temp_df = pd.read_csv(CSV_LOG_FILE, usecols=['match_id'])
            if len(temp_df) > 0:
                return int(temp_df['match_id'].max()) + 1
        except Exception:
            return 1
    return 1

# 5. Ball Outcome Simulator function (Dynamic Aggression & Prior Smoothing)
def simulate_single_ball(innings_id, over_idx, ball_idx, legal_ball, cur_score, cur_wkts, cur_rr, bat_enc, bowl_enc):
    # ফেজ অনুযায়ী প্রজেক্টেড রান রেট যাতে মডেল ডট বলের ফাঁদে না পড়ে
    expected_rr = 7.6 if over_idx < 6 else (8.0 if over_idx < 15 else 9.6)
    effective_crr = (0.6 * cur_rr + 0.4 * expected_rr) if legal_ball > 6 else expected_rr
    projected_score = max(cur_score, int(over_idx * 7.0))
    
    x_input = pd.DataFrame([[
        innings_id, over_idx, ball_idx, legal_ball, 
        projected_score, cur_wkts, effective_crr, bat_enc, bowl_enc
    ]], columns=feature_names)
    
    probs = model.predict_proba(x_input)[0]
    
    # টেম্পারেচার স্কেলিং: ডট বল কমাবে এবং স্বাভাবিক ৪, ৬ ও উইকেট নিশ্চিত করবে
    probs = np.power(probs, 0.78)
    probs = probs / probs.sum()
    
    outcome_idx = np.random.choice(len(probs), p=probs)
    return reverse_map[outcome_idx]

# 6. UI Layout - Tabs
tab_live, tab_history, tab_squads = st.tabs(["🎮 Live Match Simulation", "📜 Match History & Database", "👥 16 Teams Playing XI"])

with tab_live:
    st.sidebar.header("⚙️ Match Configuration")
    team1 = st.sidebar.selectbox("Batting First Team", teams, index=teams.index("(M) India"))
    team2_options = [t for t in teams if t != team1]
    team2 = st.sidebar.selectbox("Bowling First Team", team2_options, index=team2_options.index("(M) Australia") if "(M) Australia" in team2_options else 0)
    
    venue_selected = st.sidebar.text_input("Venue", value="Melbourne Cricket Ground")
    speed = st.sidebar.slider("Simulation Delay (Seconds per Ball)", min_value=0.0, max_value=2.0, value=0.4, step=0.1)
    start_btn = st.sidebar.button("🚀 Start Match Simulation", type="primary")

    if start_btn:
        current_match_id = get_next_match_id()
        enc_input_1 = pd.DataFrame([[team1, team2]], columns=['batting_team', 'bowling_team'])
        enc_input_2 = pd.DataFrame([[team2, team1]], columns=['batting_team', 'bowling_team'])
        
        b_enc_1, bowl_enc_1 = encoder.transform(enc_input_1)[0]
        b_enc_2, bowl_enc_2 = encoder.transform(enc_input_2)[0]
        
        team1_xi = OFFICIAL_PLAYING_XIS[team1]
        team2_xi = OFFICIAL_PLAYING_XIS[team2]
        
        # দলের মূল বোলার পুল
        t1_bowlers = [team1_xi[i] for i in [5, 6, 7, 8, 9, 10]]
        t2_bowlers = [team2_xi[i] for i in [5, 6, 7, 8, 9, 10]]
        
        st.subheader(f"🏟️ Match #{current_match_id}: {team1} vs {team2} | Venue: {venue_selected}")
        
        col1, col2 = st.columns(2)
        score_box1 = col1.empty()
        comm_box1 = col1.empty()
        score_box2 = col2.empty()
        comm_box2 = col2.empty()
        
        match_ball_records = []
        
        # ==================== 1ST INNINGS ====================
        st.info(f"▶️ 1st Innings: {team1} Batting")
        score1, wkts1, legal_balls1 = 0, 0, 0
        striker_idx1 = 0
        non_striker_idx1 = 1
        next_batter_idx1 = 2
        recent_balls1 = []
        
        for over in range(20):
            current_bowler1 = t2_bowlers[over % len(t2_bowlers)]
            
            for ball in range(1, 7):
                legal_balls1 += 1
                crr1 = ((score1 + 7.5) / (legal_balls1 + 6.0)) * 6.0
                
                outcome = simulate_single_ball(1, over, ball, legal_balls1, score1, wkts1, crr1, b_enc_1, bowl_enc_1)
                
                is_wkt = 1 if outcome == 'W' else 0
                runs_off_bat = 0 if outcome == 'W' else int(outcome)
                
                cur_striker = team1_xi[striker_idx1]
                cur_non_striker = team1_xi[non_striker_idx1]
                
                if is_wkt:
                    wkts1 += 1
                    wkt_type = np.random.choice(['caught', 'bowled', 'lbw'], p=[0.65, 0.23, 0.12])
                    player_out = cur_striker
                    desc = f"🔴 **WICKET!** {cur_striker} {wkt_type} b {current_bowler1}"
                    if next_batter_idx1 < 11:
                        striker_idx1 = next_batter_idx1
                        next_batter_idx1 += 1
                else:
                    wkt_type = None
                    player_out = None
                    score1 += runs_off_bat
                    if outcome == '6':
                        desc = f"🔥 **SIX!** {cur_striker} sends it over the ropes off {current_bowler1}!"
                    elif outcome == '4':
                        desc = f"⚡ **FOUR!** Beautiful boundary by {cur_striker}!"
                    elif outcome == '0':
                        desc = f"Dot ball to {cur_striker}"
                    else:
                        desc = f"{runs_off_bat} run(s) taken by {cur_striker}"
                        
                    if runs_off_bat in [1, 3]:
                        striker_idx1, non_striker_idx1 = non_striker_idx1, striker_idx1
                        
                recent_balls1.insert(0, f"Over {over}.{ball} ({current_bowler1}): {desc}")
                
                # মূল ডেটাসেটের ২১টি কলামের স্ট্রাকচার
                match_ball_records.append({
                    'match_id': current_match_id,
                    'venue': venue_selected,
                    'innings': 1,
                    'batting_team': team1,
                    'bowling_team': team2,
                    'over': over,
                    'ball_in_over': ball,
                    'legal_ball_number': legal_balls1,
                    'batter': cur_striker,
                    'non_striker': cur_non_striker,
                    'bowler': current_bowler1,
                    'runs_off_bat': runs_off_bat,
                    'extras': 0,
                    'total_runs': runs_off_bat,
                    'outcome': outcome,
                    'current_score': score1,
                    'wickets_lost': wkts1,
                    'is_wicket': is_wkt,
                    'wicket_type': wkt_type,
                    'player_out': player_out,
                    'current_run_rate': round((score1 / legal_balls1) * 6.0, 2)
                })
                
                score_box1.metric(
                    label=f"{team1} Innings",
                    value=f"{score1}/{wkts1} ({over}.{ball} Ov)",
                    delta=f"CRR: {round((score1 / legal_balls1) * 6.0, 2)}"
                )
                comm_box1.markdown("\n".join(recent_balls1[:5]))
                
                if wkts1 == 10:
                    break
                if speed > 0:
                    time.sleep(speed)
                    
            striker_idx1, non_striker_idx1 = non_striker_idx1, striker_idx1
            if wkts1 == 10:
                break
                
        target = score1 + 1
        st.success(f"🎯 Innings Break! {team1}: {score1}/{wkts1}. Target for {team2}: {target} runs.")
        if speed > 0:
            time.sleep(1.0)
            
        # ==================== 2ND INNINGS ====================
        st.info(f"▶️ 2nd Innings: {team2} Chasing {target} runs")
        score2, wkts2, legal_balls2 = 0, 0, 0
        striker_idx2 = 0
        non_striker_idx2 = 1
        next_batter_idx2 = 2
        recent_balls2 = []
        chase_completed = False
        
        for over in range(20):
            current_bowler2 = t1_bowlers[over % len(t1_bowlers)]
            
            for ball in range(1, 7):
                legal_balls2 += 1
                crr2 = ((score2 + 7.5) / (legal_balls2 + 6.0)) * 6.0
                
                outcome = simulate_single_ball(2, over, ball, legal_balls2, score2, wkts2, crr2, b_enc_2, bowl_enc_2)
                
                is_wkt = 1 if outcome == 'W' else 0
                runs_off_bat = 0 if outcome == 'W' else int(outcome)
                
                cur_striker = team2_xi[striker_idx2]
                cur_non_striker = team2_xi[non_striker_idx2]
                
                if is_wkt:
                    wkts2 += 1
                    wkt_type = np.random.choice(['caught', 'bowled', 'lbw'], p=[0.65, 0.23, 0.12])
                    player_out = cur_striker
                    desc = f"🔴 **WICKET!** {cur_striker} {wkt_type} b {current_bowler2}"
                    if next_batter_idx2 < 11:
                        striker_idx2 = next_batter_idx2
                        next_batter_idx2 += 1
                else:
                    wkt_type = None
                    player_out = None
                    score2 += runs_off_bat
                    if outcome == '6':
                        desc = f"🔥 **SIX!** Massive blow by {cur_striker} off {current_bowler2}!"
                    elif outcome == '4':
                        desc = f"⚡ **FOUR!** {cur_striker} punches through the gap for four!"
                    elif outcome == '0':
                        desc = f"Dot ball to {cur_striker}"
                    else:
                        desc = f"{runs_off_bat} run(s) taken by {cur_striker}"
                        
                    if runs_off_bat in [1, 3]:
                        striker_idx2, non_striker_idx2 = non_striker_idx2, striker_idx2
                        
                recent_balls2.insert(0, f"Over {over}.{ball} ({current_bowler2}): {desc}")
                
                match_ball_records.append({
                    'match_id': current_match_id,
                    'venue': venue_selected,
                    'innings': 2,
                    'batting_team': team2,
                    'bowling_team': team1,
                    'over': over,
                    'ball_in_over': ball,
                    'legal_ball_number': legal_balls2,
                    'batter': cur_striker,
                    'non_striker': cur_non_striker,
                    'bowler': current_bowler2,
                    'runs_off_bat': runs_off_bat,
                    'extras': 0,
                    'total_runs': runs_off_bat,
                    'outcome': outcome,
                    'current_score': score2,
                    'wickets_lost': wkts2,
                    'is_wicket': is_wkt,
                    'wicket_type': wkt_type,
                    'player_out': player_out,
                    'current_run_rate': round((score2 / legal_balls2) * 6.0, 2)
                })
                
                req_runs = max(0, target - score2)
                rem_balls = 120 - legal_balls2
                score_box2.metric(
                    label=f"{team2} Chase",
                    value=f"{score2}/{wkts2} ({over}.{ball} Ov)",
                    delta=f"Need {req_runs} runs off {rem_balls} balls"
                )
                comm_box2.markdown("\n".join(recent_balls2[:5]))
                
                # টার্গেট পূরণ হলে তাৎক্ষণিক ইনিংস সমাপ্তি
                if score2 >= target:
                    chase_completed = True
                    break
                if wkts2 == 10:
                    break
                if speed > 0:
                    time.sleep(speed)
                    
            striker_idx2, non_striker_idx2 = non_striker_idx2, striker_idx2
            if chase_completed or wkts2 == 10:
                break
                
        # ==================== MATCH RESULT ====================
        st.markdown("---")
        if score2 >= target:
            st.balloons()
            st.success(f"🏆 **{team2} won by {10 - wkts2} wickets!** ({score2}/{wkts2} in {over}.{ball} ov)")
        elif score1 > score2:
            st.success(f"🏆 **{team1} won by {score1 - score2} runs!** ({team2} stopped at {score2}/{wkts2})")
        else:
            st.warning("🤝 **Match Tied! Super Over Needed!**")
            
        # ==================== SAVE TO CSV ====================
        new_match_df = pd.DataFrame(match_ball_records)
        if not os.path.exists(CSV_LOG_FILE):
            new_match_df.to_csv(CSV_LOG_FILE, index=False)
        else:
            new_match_df.to_csv(CSV_LOG_FILE, mode='a', header=False, index=False)
            
        st.toast(f"Match #{current_match_id} saved successfully to {CSV_LOG_FILE}!")

# 7. Tab for Match History & CSV Explorer
with tab_history:
    st.subheader("📋 Previous Matches Database")
    
    if os.path.exists(CSV_LOG_FILE):
        full_df = pd.read_csv(CSV_LOG_FILE)
        
        matches_summary = []
        for m_id, m_data in full_df.groupby('match_id'):
            inn1 = m_data[m_data['innings'] == 1]
            inn2 = m_data[m_data['innings'] == 2]
            
            t1 = inn1['batting_team'].iloc[0]
            v_name = inn1['venue'].iloc[0]
            s1 = inn1['current_score'].iloc[-1]
            w1 = inn1['wickets_lost'].iloc[-1]
            o1 = f"{inn1['over'].iloc[-1]}.{inn1['ball_in_over'].iloc[-1]}"
            
            if len(inn2) > 0:
                t2 = inn2['batting_team'].iloc[0]
                s2 = inn2['current_score'].iloc[-1]
                w2 = inn2['wickets_lost'].iloc[-1]
                o2 = f"{inn2['over'].iloc[-1]}.{inn2['ball_in_over'].iloc[-1]}"
                
                if s2 > s1:
                    winner = f"{t2} (by {10 - w2} wkts)"
                elif s1 > s2:
                    winner = f"{t1} (by {s1 - s2} runs)"
                else:
                    winner = "Tied"
            else:
                t2 = inn1['bowling_team'].iloc[0]
                s2, w2, o2 = 0, 0, "0.0"
                winner = "Incomplete"
                
            matches_summary.append({
                'Match ID': m_id,
                'Venue': v_name,
                'Team 1': t1,
                'Score 1': f"{s1}/{w1} ({o1})",
                'Team 2': t2,
                'Score 2': f"{s2}/{w2} ({o2})",
                'Result': winner,
                'Total Balls': len(m_data)
            })
            
        summary_df = pd.DataFrame(matches_summary)
        st.dataframe(summary_df, width='stretch')
        
        # নির্দিষ্ট ম্যাচের বল-বাই-বল বিবরণী
        st.markdown("### 🔍 Specific Match Ball-by-Ball Inspection (Exact CSV Columns)")
        selected_id = st.selectbox("Select Match ID to view details", summary_df['Match ID'].tolist())
        selected_data = full_df[full_df['match_id'] == selected_id]
        
        st.dataframe(selected_data, width='stretch')
        
        # সম্পূর্ণ CSV ডাউনলোড
        csv_file_bytes = full_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Ball-by-Ball CSV",
            data=csv_file_bytes,
            file_name=CSV_LOG_FILE,
            mime='text/csv'
        )
    else:
        st.info("এখনো কোনো ম্যাচ খেলা হয়নি। প্রথম ট্যাবে গিয়ে ম্যাচ শুরু করুন!")

# 8. Tab for Viewing 16 Teams Playing XI
with tab_squads:
    st.subheader("👥 All 16 Teams Historical Playing XI (All-Time Elite Rosters)")
    selected_team = st.selectbox("Select Team to view Playing XI", teams)
    
    xi_list = OFFICIAL_PLAYING_XIS[selected_team]
    xi_df = pd.DataFrame({
        "Batting Position": [f"#{i+1}" for i in range(11)],
        "Player Name": xi_list,
        "Role": [
            "Top Order Batter" if i < 3 else 
            "Middle Order Batter / WK" if i < 5 else 
            "All-Rounder" if i < 7 else 
            "Main Specialist Bowler" for i in range(11)
        ]
    })
    st.table(xi_df)
