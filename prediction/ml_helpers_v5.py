"""ml_helpers_v5.py — v5 prediction pipeline with roles, reasons, improvement, career paths."""
import os, numpy as np, joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')

FEATURE_COLS = [
    'age', 'gender',
    'likes_sports', 'sport_indoor_outdoor', 'which_sport',
    'sport_hours_per_day', 'sport_team_solo', 'sport_activity_level', 'sport_frequency',
    'likes_arts', 'which_art', 'art_sub_type',
    'art_creativity', 'art_performance', 'art_hours',
    'art_learning_style', 'art_digital_traditional', 'art_with_whom',
    'dance_style', 'music_type', 'acting_interest', 'performance_stage_comfort',
    'likes_academics', 'fav_subject', 'acad_problem_solving',
    'acad_curiosity', 'acad_study_type', 'acad_competitions', 'acad_reading_habit',
    'likes_analytical', 'analy_puzzle_type', 'analy_logic_level',
    'analy_challenge', 'analy_work_style', 'analy_coding_interest', 'analy_patience_level',
    'health_condition', 'health_energy', 'health_outdoor_preference',
    'health_activity_preference', 'health_hours', 'health_sleep_quality',
    'likes_cooking', 'cooking_type',
    'likes_gardening', 'gardening_type', 'nature_interest',
    'screen_usage_hours', 'screen_content_type', 'gaming_genre',
    'game_design_interest', 'digital_creation_interest',
    'activity_with_whom', 'activity_time_of_day',
    'emotional_engagement', 'initiative_level',
    'tried_hobby_before', 'previous_hobby_name', 'stopped_reason',
    'personality_type', 'creativity_score', 'leadership_tendency', 'attention_span',
]
NUMERIC_COLS = {'age', 'creativity_score'}

HEALTH_RESTRICTIONS = {
    'Asthma': {
        'blocked': ['Cricket', 'Football', 'Running', 'Athletics'],
        'safe_alt': 'Swimming',
        'warning': 'Due to Asthma, high-intensity outdoor activities may trigger breathing difficulties. Swimming and indoor activities are safer alternatives.',
    },
    'Joint Pain': {
        'blocked': ['Gymnastics', 'Running', 'Athletics'],
        'safe_alt': 'Yoga',
        'warning': 'Due to Joint Pain, high-impact activities can worsen the condition. Yoga and Swimming are gentle, low-impact alternatives.',
    },
    'Autism': {
        'blocked': [],
        'safe_alt': None,
        'warning': 'Activities with clear structure, predictable rules, and calm environments are recommended. Chess, Drawing, and Gardening work especially well.',
        'preferred': ['Chess', 'Drawing', 'Gardening', 'Meditation', 'Robotics'],
    },
    'Physical Limitation': {
        'blocked': ['Cricket', 'Football', 'Running', 'Athletics', 'Gymnastics'],
        'safe_alt': 'Drawing',
        'warning': 'Physical activities with high impact are not recommended. Creative, digital, and calm activities are ideal.',
    },
}

HOBBY_META = {
    'Cricket':          {'category':'Sports','icon':'🏏','desc':'A team outdoor sport focusing on strategy and physical fitness','outdoor':True},
    'Football':         {'category':'Sports','icon':'⚽','desc':'A high-energy team sport building endurance and teamwork','outdoor':True},
    'Badminton':        {'category':'Sports','icon':'🏸','desc':'A fast-paced racket sport improving reflexes and agility','outdoor':False},
    'Swimming':         {'category':'Sports','icon':'🏊','desc':'A full-body workout great for fitness and relaxation','outdoor':False},
    'Table Tennis':     {'category':'Sports','icon':'🏓','desc':'An indoor precision sport enhancing focus and coordination','outdoor':False},
    'Basketball':       {'category':'Sports','icon':'🏀','desc':'A high-energy team sport building coordination and teamwork','outdoor':True},
    'Athletics':        {'category':'Sports','icon':'🏅','desc':'Track and field sports building speed and endurance','outdoor':True},
    'Carrom':           {'category':'Sports','icon':'🎯','desc':'A strategic indoor board game improving precision','outdoor':False},
    'Drawing':          {'category':'Arts','icon':'🎨','desc':'Express creativity through sketching and visual art','outdoor':False},
    'Dance':            {'category':'Arts','icon':'💃','desc':'Physical expression through rhythm and movement','outdoor':False},
    'Music':            {'category':'Arts','icon':'🎵','desc':'Learn instruments or vocals to express through sound','outdoor':False},
    'Acting':           {'category':'Arts','icon':'🎭','desc':'Bring stories to life through performance and expression','outdoor':False},
    'Math Olympiad':    {'category':'Academics','icon':'🔢','desc':'Competitive mathematics building logical reasoning','outdoor':False},
    'Science Club':     {'category':'Academics','icon':'🔬','desc':'Explore science through experiments and group projects','outdoor':False},
    'Creative Writing': {'category':'Academics','icon':'✍️','desc':'Develop storytelling and language skills through writing','outdoor':False},
    'Coding':           {'category':'Analytical','icon':'💻','desc':'Build programs and solve problems with code','outdoor':False},
    'Chess':            {'category':'Analytical','icon':'♟️','desc':'Strategic board game developing critical thinking','outdoor':False},
    'Robotics':         {'category':'Analytical','icon':'🤖','desc':'Build and program robots combining engineering and coding','outdoor':False},
    'Yoga':             {'category':'Health','icon':'🧘','desc':'Mind-body practice improving flexibility and calm','outdoor':False},
    'Gymnastics':       {'category':'Health','icon':'🤸','desc':'Develop strength, flexibility, and body control','outdoor':False},
    'Running':          {'category':'Health','icon':'🏃','desc':'Build endurance and cardiovascular fitness','outdoor':True},
    'Meditation':       {'category':'Health','icon':'🌿','desc':'Calm the mind and build focus through mindfulness','outdoor':False},
    'Baking':           {'category':'Cooking','icon':'🍰','desc':'Create delicious baked goods through creativity and precision','outdoor':False},
    'Cooking Meals':    {'category':'Cooking','icon':'👨‍🍳','desc':'Prepare nutritious meals developing culinary creativity','outdoor':False},
    'Gardening':        {'category':'Gardening','icon':'🌱','desc':'Nurture plants and connect with nature','outdoor':True},
    'Game Design':      {'category':'Digital','icon':'🎮','desc':'Design and build interactive games using creativity and logic','outdoor':False},
    'Digital Content':  {'category':'Digital','icon':'📱','desc':'Create videos, art, and content for digital platforms','outdoor':False},
}

HOBBY_ROLES = {
    'Cricket':          ['Competitive Player','Team Captain','Cricket Coach'],
    'Football':         ['Striker','Team Player','Football Coach'],
    'Badminton':        ['Competitive Player','Club Player','Badminton Coach'],
    'Swimming':         ['Competitive Swimmer','Recreational Swimmer','Swimming Coach'],
    'Basketball':       ['Point Guard','Team Captain','Basketball Coach'],
    'Athletics':        ['Sprinter','Long-distance Runner','Athletics Coach'],
    'Chess':            ['Competitive Player','Online Chess Coach','Tournament Organizer'],
    'Drawing':          ['Hobby Artist','Commercial Artist','Graphic Designer'],
    'Dance':            ['Performer','Choreographer','Dance Teacher'],
    'Music':            ['Vocalist','Instrumentalist','Music Composer'],
    'Acting':           ['Stage Actor','Film Actor','Drama Director'],
    'Coding':           ['App Developer','Game Developer','Web Developer'],
    'Robotics':         ['Robot Builder','AI Programmer','Robotics Engineer'],
    'Math Olympiad':    ['Olympiad Competitor','Math Tutor','Researcher'],
    'Science Club':     ['Experimenter','Science Communicator','Researcher'],
    'Creative Writing': ['Fiction Writer','Blogger/Journalist','Poet'],
    'Yoga':             ['Yoga Practitioner','Yoga Instructor','Wellness Coach'],
    'Gymnastics':       ['Gymnast','Gymnastics Coach','Fitness Trainer'],
    'Meditation':       ['Mindfulness Practitioner','Mindfulness Guide'],
    'Running':          ['Casual Runner','Competitive Runner','Marathon Athlete'],
    'Baking':           ['Home Baker','Pastry Chef','Baking Instructor'],
    'Cooking Meals':    ['Home Cook','Professional Chef','Food Blogger'],
    'Gardening':        ['Home Gardener','Landscape Designer','Botanist'],
    'Game Design':      ['Game Designer','Game Developer','Indie Game Creator'],
    'Digital Content':  ['Content Creator','YouTuber','Digital Artist'],
}

CAREER_PATHS = {
    'Cricket':          [{'title':'Professional Cricketer','path':'School Team → District → State → IPL/National'},{'title':'Cricket Coach','path':'Playing career → Coaching certification → Academy'},{'title':'Cricket Commentator','path':'Deep game knowledge → Broadcasting → Media career'}],
    'Football':         [{'title':'Professional Footballer','path':'Youth Club → State League → National Team'},{'title':'Football Coach','path':'Playing career → UEFA/AFC License → Club Coach'},{'title':'Sports Manager','path':'Sports Management degree → Club Administration'}],
    'Badminton':        [{'title':'Professional Badminton Player','path':'Academy → BWF Circuit → International Tournaments'},{'title':'Badminton Coach','path':'State level play → Coaching certification → Academy'}],
    'Swimming':         [{'title':'Competitive Swimmer','path':'Swimming Club → National Championships → Olympics'},{'title':'Swimming Coach','path':'Competitive career → Coaching certification → Academy'},{'title':'Lifeguard/Water Safety','path':'Certification → Aquatic Centers → Water Safety Instructor'}],
    'Basketball':       [{'title':'Professional Basketball Player','path':'School → State → National League'},{'title':'Basketball Coach','path':'Playing career → Coaching → Club or School'}],
    'Chess':            [{'title':'Competitive Chess Player','path':'School Tournaments → FIDE Rating → National/International'},{'title':'Online Chess Coach','path':'Strong FIDE rating → Chess.com/Lichess coaching → Online Academy'},{'title':'Chess Educator','path':'Teaching chess in schools → Chess curriculum development'}],
    'Drawing':          [{'title':'Graphic Designer','path':'Art school → Design software skills → Agency or Freelance'},{'title':'Illustrator/Animator','path':'Portfolio building → Art courses → Studio or Freelance'},{'title':'Fine Artist','path':'Art exhibitions → Galleries → Commission-based career'}],
    'Dance':            [{'title':'Professional Dancer','path':'Dance Academy → Competitions → Performance Troupe'},{'title':'Choreographer','path':'Performance experience → Choreography training → Own productions'},{'title':'Dance Teacher','path':'Certification → Academy or Own Studio'}],
    'Music':            [{'title':'Playback Singer/Performer','path':'Music Academy → Competitions → Recording career'},{'title':'Music Composer','path':'Music theory → DAW skills → Film/Game scoring'},{'title':'Music Teacher','path':'Degree in Music → Teaching certification → School or Studio'}],
    'Acting':           [{'title':'Theatre/Film Actor','path':'Drama school → Auditions → Stage/Film career'},{'title':'Drama Teacher','path':'Theatre experience → Teaching degree → School Drama'},{'title':'Director','path':'Acting experience → Directing workshops → Own productions'}],
    'Coding':           [{'title':'Software Engineer','path':'Coding clubs → CS degree → Tech company or Startup'},{'title':'Game Developer','path':'Game engines (Unity/Godot) → Portfolio → Game Studio'},{'title':'Startup Founder','path':'Side projects → MVP → Entrepreneurship'}],
    'Robotics':         [{'title':'Robotics Engineer','path':'Robotics competitions → Engineering degree → Industry'},{'title':'AI/ML Researcher','path':'Coding + Math skills → CS/AI degree → Research lab'},{'title':'Mechatronics Engineer','path':'Robotics club → Engineering college → Manufacturing'}],
    'Math Olympiad':    [{'title':'Data Scientist','path':'Math Olympiad → Statistics/CS degree → Tech industry'},{'title':'Quantitative Analyst','path':'Strong math → Finance degree → Investment banks'},{'title':'Professor/Researcher','path':'Olympiad → Top university → Academia'}],
    'Science Club':     [{'title':'Research Scientist','path':'Science Olympiad → Science degree → Research lab'},{'title':'Engineer','path':'Science interest → Engineering college → Industry'},{'title':'Science Communicator','path':'Science knowledge → Writing/Media → YouTube/Journalism'}],
    'Creative Writing': [{'title':'Author','path':'Writing practice → Literary agents → Published books'},{'title':'Content Creator','path':'Blog/Channel → Audience building → Brand deals'},{'title':'Journalist/Editor','path':'Writing skills → Journalism degree → Publications'}],
    'Yoga':             [{'title':'Yoga Instructor','path':'Daily practice → Yoga certification → Studio or Online classes'},{'title':'Wellness Coach','path':'Yoga + nutrition knowledge → Wellness certification → Own practice'}],
    'Gymnastics':       [{'title':'Olympic Gymnast','path':'Gymnastics club → National Championships → Olympic trials'},{'title':'Gymnastics Coach','path':'Competitive career → Coaching certification → Academy'}],
    'Meditation':       [{'title':'Mindfulness Coach','path':'Practice depth → Mindfulness certification → Corporate/Therapy'},{'title':'Therapist','path':'Psychology degree + mindfulness → Clinical practice'}],
    'Baking':           [{'title':'Pastry Chef','path':'Home baking → Culinary school → Hotel or Bakery'},{'title':'Bakery Owner','path':'Recipe mastery → Business skills → Own bakery'},{'title':'Food Blogger/YouTuber','path':'Recipe content → Audience → Brand partnerships'}],
    'Cooking Meals':    [{'title':'Professional Chef','path':'Home cooking → Culinary institute → Restaurant'},{'title':'Restaurant Owner','path':'Chef skills → Business management → Own restaurant'},{'title':'Food Content Creator','path':'Cooking videos → Social media → Brand deals'}],
    'Gardening':        [{'title':'Botanist','path':'Gardening passion → Botany degree → Research/Conservation'},{'title':'Landscape Architect','path':'Garden design → Architecture degree → Firm or Freelance'},{'title':'Organic Farmer','path':'Gardening skills → Agriculture knowledge → Own farm'}],
    'Game Design':      [{'title':'Game Designer','path':'Game design tools → Portfolio → Game studio'},{'title':'Indie Game Developer','path':'Game engines → Publish on itch.io/Steam → Indie career'},{'title':'Esports Professional','path':'Competitive gaming → Tournament circuit → Team contract'}],
    'Digital Content':  [{'title':'Content Creator/YouTuber','path':'Create content → Build audience → Monetisation'},{'title':'Digital Marketer','path':'Content skills → Marketing courses → Agency or Brand'},{'title':'Digital Artist','path':'Digital art portfolio → Freelance platforms → Commission work'}],
}

IMPROVEMENT_TIPS = {
    'Cricket':          ['Practice batting or bowling for 30 min daily','Join a local cricket academy or club','Study match strategies by watching professional games','Work on fitness: running and agility drills'],
    'Football':         ['Dribble and shoot for 20 min daily','Join a youth football club for team play','Practice footwork: cones and ladder drills','Watch professional matches and study positions'],
    'Badminton':        ['Practice footwork and smash daily','Join a badminton academy for technique training','Learn backhand and net play progressively','Play practice matches to build game awareness'],
    'Swimming':         ['Swim 3-4 days per week consistently','Focus on breathing technique and stroke form','Try different strokes: freestyle, backstroke, butterfly','Work with a coach for technique correction'],
    'Chess':            ['Solve 3 chess puzzles daily on Chess.com or Lichess','Study one opening system per month','Play rated games online to track Elo progress','Review your lost games to identify mistakes'],
    'Drawing':          ['Sketch daily for 20 minutes without pressure','Study one art technique or style per week','Join an art class or online workshop','Build a portfolio of your best work'],
    'Dance':            ['Practice choreography for 30 minutes daily','Join a local dance academy for structured training','Learn a new dance style every 3 months','Watch performances to study technique and expression'],
    'Music':            ['Practice your instrument for 20-30 minutes daily','Learn music theory alongside playing','Perform for family monthly to build confidence','Record yourself to identify areas to improve'],
    'Acting':           ['Join a school or local drama club','Practice monologues daily in front of a mirror','Study famous performances and analyse techniques','Participate in school plays and community theatre'],
    'Coding':           ['Code for 30 minutes daily on Scratch, Python, or JavaScript','Complete one small project per month','Participate in school or online coding competitions','Learn one new concept each week from tutorials'],
    'Robotics':         ['Join a school or community robotics club','Build simple circuits and Arduino projects at home','Participate in robotics competitions like First Lego League','Learn basic electronics alongside programming'],
    'Math Olympiad':    ['Solve 5 olympiad-level problems daily','Join a math enrichment class or club','Practice past year olympiad papers regularly','Form a study group with similarly motivated peers'],
    'Science Club':     ['Conduct one experiment weekly at home','Read science magazines like National Geographic Kids','Join a science fair and submit a project','Watch science YouTube channels for inspiration'],
    'Creative Writing': ['Write at least 1 page daily — any topic','Read across diverse genres to expand vocabulary','Share stories with family or a writing group for feedback','Enter school or national writing competitions'],
    'Yoga':             ['Practice yoga for 15-20 minutes every morning','Learn 5 new asanas per month','Meditate for 5 minutes after each yoga session','Try breathing exercises (pranayama) for focus'],
    'Gymnastics':       ['Stretch and condition daily for flexibility','Work with a certified gymnastics coach','Practice basic floor routines progressively','Set milestone goals: cartwheel, handstand, backflip'],
    'Meditation':       ['Meditate for 10 minutes each morning','Use guided meditation apps like Headspace or Calm','Practice mindful breathing during stressful moments','Try body scan and visualization techniques'],
    'Baking':           ['Bake one new recipe every week','Learn decorating techniques: icing, piping, fondant','Experiment with healthy substitutes in recipes','Watch baking tutorials and try replicating them'],
    'Cooking Meals':    ['Cook one new recipe every week','Learn basic knife skills and kitchen safety','Watch cooking tutorials and try different cuisines','Keep a recipe journal of creations and improvements'],
    'Gardening':        ['Tend to plants every day — watering and observation','Start with easy herbs: mint, basil, coriander','Learn about soil types and composting','Keep a garden journal to track plant growth'],
    'Game Design':      ['Use Scratch, GDevelop, or Unity to build small games','Study game mechanics by analysing your favourite games','Complete and publish one small game per month','Join game jam events like Ludum Dare for practice'],
    'Digital Content':  ['Create and post content consistently every week','Learn basic video editing with DaVinci Resolve or CapCut','Study your audience and improve based on feedback','Collaborate with peers to grow your channel faster'],
    'Running':          ['Run 3-4 times per week, gradually increasing distance','Focus on proper running form and breathing','Track progress with a running app','Enter local fun runs or school athletics events'],
}

HOBBY_RULES = {
    'Sports':     {'key':'which_sport',               'default':'Cricket'},
    'Arts':       {'key':'which_art',                 'default':'Drawing'},
    'Analytical': {'key':'analy_puzzle_type',         'default':'Chess',
                   'map':{'Coding':'Robotics','Puzzles':'Chess','Logical Games':'Robotics','Robotics':'Robotics'}},
    'Health':     {'key':'health_activity_preference','default':'Yoga'},
    'Cooking':    {'key':'cooking_type',              'default':'Baking'},
    'Gardening':  {'key':'gardening_type',            'default':'Gardening', 'constant':'Gardening'},
    'Digital':    {'key':'game_design_interest',      'default':'Digital Content'},
}


def load_model():
    for v in ['v5','v4','v3']:
        try:
            m   = joblib.load(os.path.join(SAVE_DIR, f'model_{v}.pkl'))
            le  = joblib.load(os.path.join(SAVE_DIR, f'label_encoders_{v}.pkl'))
            te  = joblib.load(os.path.join(SAVE_DIR, f'target_encoder_{v}.pkl'))
            # Load the actual feature columns the model was trained on
            try:
                fc = joblib.load(os.path.join(SAVE_DIR, f'feature_cols_{v}.pkl'))
            except Exception:
                fc = FEATURE_COLS  # fallback to hardcoded list
            print(f'[ml_helpers_v5] Loaded {v} model with {len(fc)} features')
            return m, le, te, fc
        except Exception:
            pass
    return None, None, None, FEATURE_COLS


def build_feature_vector(answers, label_encs, feature_cols=None):
    """Build feature vector using the exact columns the model was trained on."""
    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    encoded = []
    for col in cols:
        raw = answers.get(col, 'None')
        if col in NUMERIC_COLS:
            try:    encoded.append(float(raw))
            except: encoded.append(8.0 if col == 'age' else 5.0)
            continue
        le = label_encs.get(col)
        if le is None:
            encoded.append(0.0); continue
        s = str(raw)
        encoded.append(float(le.transform([s])[0]) if s in le.classes_ else 0.0)
    return np.array([encoded], dtype=float)


def pick_hobby(category, answers):
    rule = HOBBY_RULES.get(category, {})
    if rule.get('constant'):
        return rule['constant']
    key = rule.get('key', '')
    val = str(answers.get(key, 'None'))
    if 'map' in rule:
        return rule['map'].get(val, rule['default'])
    if category == 'Digital':
        return 'Game Design' if answers.get('game_design_interest') in ('Yes', 'High', 'Medium') else 'Digital Content'
    art_map = {'Drawing':'Drawing','Dance':'Dance','Music':'Music','Acting':'Acting'}
    if category == 'Arts':
        return art_map.get(val, rule.get('default','Drawing'))
    return val if val not in ('None','Other','') else rule.get('default', category)


def pick_role(hobby, answers):
    roles = HOBBY_ROLES.get(hobby, ['Practitioner','Enthusiast'])
    a = answers
    if hobby == 'Dance':
        stage = a.get('performance_stage_comfort','')
        creat = a.get('art_creativity','')
        if stage == 'High':   return 'Performer'
        if creat == 'High':   return 'Choreographer'
        return 'Dance Teacher'
    if hobby == 'Game Design':
        return 'Game Developer' if a.get('analy_coding_interest') in ('Yes', 'High', 'Medium') else 'Game Designer'
    if hobby == 'Drawing':
        return 'Commercial Artist' if a.get('art_performance') in ('Yes', 'High', 'Medium') else 'Hobby Artist'
    if hobby == 'Music':
        mt = a.get('music_type','')
        if mt == 'Vocals':     return 'Vocalist'
        if mt == 'Instrument': return 'Instrumentalist'
        return 'Music Composer'
    if hobby == 'Coding':
        sc = a.get('screen_content_type','')
        if sc == 'Gaming' and a.get('game_design_interest') in ('Yes', 'High', 'Medium'): return 'Game Developer'
        return 'App Developer'
    if hobby in ('Cricket','Football','Basketball','Badminton','Swimming'):
        return 'Team Captain' if a.get('sport_team_solo') == 'Team' and a.get('sport_activity_level') == 'High' else roles[0]
    return roles[0]


def apply_health_filter(hobby, health_cond, answers):
    rest = HEALTH_RESTRICTIONS.get(health_cond)
    if not rest:
        return hobby, None
    if health_cond == 'Autism':
        preferred = rest.get('preferred', [])
        if hobby not in preferred and preferred:
            return preferred[0], rest['warning']
        return hobby, rest['warning']
    if hobby in rest.get('blocked', []):
        return rest['safe_alt'], rest['warning']
    return hobby, None


def build_reason(category, hobby, role, answers):
    """Build a deeply personalized reason by referencing the child's actual answers."""
    a = answers
    age = int(a.get('age', 8))
    gender = a.get('gender', 'the child')
    child = 'your child'

    # ── 1. Core trait sentence (based on specific answers, NOT generic category) ──
    traits = []

    # Energy + engagement
    energy = a.get('health_energy', 'Medium')
    engagement = a.get('emotional_engagement', 'Moderate')
    is_high_energy = energy in ('High', 'Very high')
    is_engaged = engagement in ('Very Engaged', 'High')
    if is_high_energy and is_engaged:
        traits.append('has high energy and gets deeply absorbed in activities')
    elif is_high_energy:
        traits.append('is very energetic and physically active')
    elif is_engaged:
        traits.append('shows deep passion and emotional connection to their interests')
    elif energy == 'Low':
        traits.append('prefers calm, focused activities over high-energy ones')

    # Social style
    whom = a.get('activity_with_whom', 'Mixed')
    team = a.get('sport_team_solo', '')
    if whom == 'Alone' or team == 'Individual':
        traits.append('works best independently with focused attention')
    elif whom == 'With Friends' or team == 'Team':
        traits.append('thrives in social settings and enjoys collaborative activities')
    elif whom == 'With Family':
        traits.append('values family time and prefers bonding through shared activities')

    # Category-specific depth
    if category == 'Sports':
        sport = a.get('which_sport', '')
        hours = a.get('sport_hours_per_day', 'Low')
        freq = a.get('sport_frequency', '1-2 days')
        if hours == 'High' or freq == '5+ days':
            traits.append(f'already dedicates significant time to {sport}' if sport and sport != 'None' else 'spends substantial time on sports')
        io = a.get('sport_indoor_outdoor', '')
        if io == 'outdoor':
            traits.append('loves outdoor environments and open-air play')
        elif io == 'indoor':
            traits.append('prefers indoor sports that require precision and focus')

    elif category == 'Arts':
        art = a.get('which_art', '')
        sub = a.get('art_sub_type', '')
        creat = a.get('art_creativity', 'Medium')
        perf = a.get('art_performance', 'Low')
        if creat == 'High':
            traits.append('demonstrates exceptional creative imagination')
        if perf in ('High', 'Medium', 'Yes'):
            traits.append('loves performing and sharing their work with others')
        elif perf in ('Low', 'No') and creat == 'High':
            traits.append('channels creativity through personal expression rather than public performance')
        if sub and sub != 'None':
            traits.append(f'has a clear preference for {sub.lower()} as their creative outlet')

    elif category == 'Analytical':
        puzzle = a.get('analy_puzzle_type', '')
        logic = a.get('analy_logic_level', 'Medium')
        patience = a.get('analy_patience_level', 'Medium')
        tech = a.get('analy_coding_interest', 'Low')
        if logic in ('High', 'Medium'):
            traits.append('has strong logical reasoning abilities')
        if patience in ('High', 'Medium'):
            traits.append('shows exceptional patience and persistence with complex challenges')
        if tech in ('High', 'Medium', 'Yes'):
            traits.append('is drawn to computers and digital technology')
        if puzzle and puzzle not in ('None', 'Other'):
            puzzle_names = {'Coding': 'technology and digital tools', 'Puzzles': 'chess and board games',
                          'Logical Games': 'strategy and logic games', 'Robotics': 'building and electronics'}
            traits.append(f'particularly enjoys {puzzle_names.get(puzzle, puzzle.lower())}')

    elif category == 'Health':
        act = a.get('health_activity_preference', '')
        sleep = a.get('health_sleep_quality', 'Average')
        if act and act not in ('None', 'Other'):
            traits.append(f'shows natural inclination towards {act.lower()} for physical wellness')
        if sleep == 'Good':
            traits.append('maintains healthy sleep habits supporting overall wellbeing')

    elif category == 'Cooking':
        cook_type = a.get('cooking_type', '')
        if cook_type == 'Baking':
            traits.append('enjoys the precision and creativity of baking')
        elif cook_type == 'Cooking Meals':
            traits.append('is interested in preparing nutritious meals and trying new recipes')
        else:
            traits.append('shows enthusiasm for both baking and cooking')

    elif category == 'Gardening':
        gard_type = a.get('gardening_type', '')
        if gard_type == 'Plants/Flowers':
            traits.append('has a nurturing side and loves caring for flowers and plants')
        elif gard_type == 'Vegetables':
            traits.append('enjoys the rewarding process of growing food from seed to harvest')
        else:
            traits.append('shows genuine connection with nature and patience in nurturing growth')

    elif category == 'Digital':
        content = a.get('screen_content_type', '')
        design = a.get('game_design_interest', 'No')
        genre = a.get('gaming_genre', '')
        if design == 'Yes':
            traits.append('wants to create digital experiences, not just consume them')
        if genre == 'Strategy':
            traits.append('prefers strategic thinking games that challenge the mind')
        elif genre == 'Puzzle':
            traits.append('gravitates toward puzzle-based digital challenges')
        if content == 'Creative Content':
            traits.append('already explores creative digital expression')

    # ── 2. Build the core reason paragraph ──
    if not traits:
        traits.append('shows strong natural alignment with this area based on their overall behaviour profile')

    # Cap at 3 most relevant traits
    core_traits = traits[:3]
    reason = f'Based on the assessment, {child} ' + ', '.join(core_traits[:-1])
    if len(core_traits) > 1:
        reason += f', and {core_traits[-1]}.'
    else:
        reason += '.'

    # ── 3. Role justification ──
    if role:
        role_reasons = {
            'Performer': f'Their love for the spotlight and stage confidence makes the "{role}" path a natural fit.',
            'Choreographer': f'Their high creativity combined with a preference for behind-the-scenes work suits the "{role}" role perfectly.',
            'Commercial Artist': f'Since they enjoy sharing their work publicly, the "{role}" path can turn their talent into opportunity.',
            'Hobby Artist': f'They create for personal joy rather than attention — the "{role}" path lets them grow at their own pace.',
            'Game Developer': f'Their coding interest combined with gaming enthusiasm opens the door to "{role}" as a creative-technical role.',
            'Game Designer': f'Their creative vision for games, even without deep coding, positions them well as a "{role}".',
            'Team Captain': f'Their team spirit and high activity level suggest natural leadership — a "{role}" in the making.',
            'Vocalist': f'Their interest in singing shows vocal expression as their primary musical outlet.',
            'Instrumentalist': f'Their focus on instruments indicates a hands-on approach to music.',
            'App Developer': f'Their coding skills and problem-solving ability point toward building real-world applications.',
        }
        role_line = role_reasons.get(role, f'The "{role}" specialisation aligns well with their specific strengths.')
        reason += f' {role_line}'

    # ── 4. Previous hobby context (specific, not generic) ──
    if a.get('tried_hobby_before') == 'Yes':
        prev = a.get('previous_hobby_name', '')
        stopped = a.get('stopped_reason', '')
        if prev and prev != 'None' and stopped and stopped != 'None':
            stop_advice = {
                'Lost Interest': f'They previously tried {prev} but lost interest. This recommendation offers a fresh direction that better matches their current behaviour patterns.',
                'Teacher/Coach Issue': f'A past experience with {prev} was affected by a teacher/coach issue. We\'ve focused on hobbies that can thrive with self-directed learning too.',
                'Environment Issue': f'Their earlier attempt at {prev} was limited by environment. This hobby is more accessible and adaptable to various settings.',
                'No Resources': f'They wanted to continue {prev} but lacked resources. This recommendation considers accessibility and low-barrier entry.',
                'Completed It': f'They successfully completed their journey with {prev} — this is a natural next step that builds on those developed skills.',
            }
            reason += f' {stop_advice.get(stopped, f"Their experience with {prev} has been considered in this recommendation.")}'

    return reason


def predict_hobby_v5(answers):
    model, label_encs, target_enc, feature_cols = load_model()
    if model is None:
        return None

    health_cond = str(answers.get('health_condition','None'))
    all_classes = list(target_enc.classes_)

    vec   = build_feature_vector(answers, label_encs, feature_cols)
    probs = model.predict_proba(vec)[0]

    # Categories removed from the questionnaire — never surface these
    REMOVED_CATEGORIES = {'Academics'}

    # Explicit preference override
    explicit_map = {
        'Sports':    answers.get('likes_sports','No') == 'Yes',
        'Arts':      answers.get('likes_arts','No')   == 'Yes',
        'Analytical':answers.get('likes_analytical','No') == 'Yes',
        'Cooking':   answers.get('likes_cooking','No') == 'Yes',
        'Gardening': answers.get('likes_gardening','No') == 'Yes',
    }
    liked = [c for c,v in explicit_map.items() if v]

    if len(liked) == 1:
        category = liked[0]
    elif liked:
        best_cat, best_prob = None, -1.0
        for cat in liked:
            if cat in all_classes:
                p = float(probs[all_classes.index(cat)])
                if p > best_prob:
                    best_prob, best_cat = p, cat
        category = best_cat or liked[0]
    else:
        # Check digital explicitly
        if answers.get('screen_usage_hours') == 'High' or answers.get('game_design_interest') == 'Yes':
            category = 'Digital'
        else:
            # Pick best category that isn't removed
            for idx in np.argsort(probs)[::-1]:
                cat = all_classes[idx]
                if cat not in REMOVED_CATEGORIES:
                    category = cat
                    break
            else:
                category = all_classes[int(np.argmax(probs))]

    cat_idx  = all_classes.index(category) if category in all_classes else 0
    conf     = round(float(probs[cat_idx]) * 100, 1)

    hobby = pick_hobby(category, answers)
    hobby, health_warning = apply_health_filter(hobby, health_cond, answers)
    role  = pick_role(hobby, answers)
    meta  = HOBBY_META.get(hobby, {})

    # Alternatives (top-3 other categories, excluding removed ones)
    sorted_idxs  = np.argsort(probs)[::-1]
    alternatives = []
    seen_cats    = {category} | REMOVED_CATEGORIES
    for idx in sorted_idxs:
        if len(alternatives) >= 3: break
        cat = all_classes[idx]
        if cat in seen_cats: continue
        seen_cats.add(cat)
        alt_h = pick_hobby(cat, answers)
        safe_alt, alt_warn = apply_health_filter(alt_h, health_cond, answers)
        alt_meta = HOBBY_META.get(safe_alt, {})
        alternatives.append({
            'hobby':         safe_alt,
            'category':      cat,
            'confidence':    round(float(probs[idx]) * 100, 1),
            'desc':          alt_meta.get('desc',''),
            'icon':          alt_meta.get('icon','⭐'),
            'health_warning':alt_warn,
        })

    return {
        'predicted_hobby':        hobby,
        'hobby_role':             role,
        'category':               meta.get('category', category),
        'confidence_score':       conf,
        'description':            meta.get('desc',''),
        'icon':                   meta.get('icon','⭐'),
        'recommendation_reason':  build_reason(category, hobby, role, answers),
        'improvement_suggestions':IMPROVEMENT_TIPS.get(hobby, ['Practice daily', 'Join a local club', 'Set monthly goals']),
        'career_paths':           CAREER_PATHS.get(hobby, [{'title':'Professional','path':'Train → Compete → Career'}]),
        'health_warning':         health_warning,
        'alternatives':           alternatives,
    }
