from flask import Flask, request, redirect, session, jsonify, render_template
import requests
import sqlitecloud
from urllib.parse import urlencode
import os
from dotenv import load_dotenv
import jwt
from functools import wraps
import numpy as np
from openai import OpenAI
import httpx
from pinecone import ServerlessSpec, Pinecone

# 載入環境變數
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')  # 必须设置
app.config['SESSION_COOKIE_SECURE'] = True  # 对于 HTTPS 是必须的
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# LINE Login 配置
LINE_LOGIN_CLIENT_ID = os.getenv('LINE_LOGIN_CLIENT_ID')
LINE_LOGIN_CLIENT_SECRET = os.getenv('LINE_LOGIN_CLIENT_SECRET')
SQLITE_CLOUD_URL = os.getenv("SQLITE_CLOUD_URL")
REDIRECT_URI = os.getenv('REDIRECT_URI')
LINE_AUTH_URL = os.getenv('LINE_AUTH_URL')
LINE_TOKEN_URL = os.getenv('LINE_TOKEN_URL')
LINE_PROFILE_URL = os.getenv('LINE_PROFILE_URL')

# Pinecone 和 OpenAI 配置
PINECONE_API_KEY = os.getenv("PINECONE_KEY")
OPENAI_API_KEY = os.getenv("GPT_KEY")

# 初始化 Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone client initialized successfully:", type(pc))
except Exception as e:
    print(f"Failed to initialize Pinecone client: {e}")
    raise

index_name = "searching"
try:
    index_list = pc.list_indexes().names()
    print("Available indexes:", index_list)
    if index_name not in index_list:
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name, 
            dimension=1536,
            metric="euclidean",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
        print(f"Index '{index_name}' created")
    search_index = pc.Index(index_name)
    print("Pinecone index loaded:", type(search_index), search_index)
except Exception as e:
    print(f"Failed to initialize Pinecone index: {e}")
    raise

# 初始化 OpenAI
client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=httpx.Client(proxies=None)
)

# 文字轉向量函數
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)

# 初始化 SQLite Cloud 資料庫
def init_db():
    print("Attempting to initialize database...")
    try:
        conn = sqlitecloud.connect(SQLITE_CLOUD_URL)
        cursor = conn.cursor()
        print("Database connection successful.")
        # 創建 line_users 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS line_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL
            )
        ''')
        print("Executed CREATE TABLE IF NOT EXISTS for line_users.")
        # 創建 user_pinned 表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_pinned (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lineid TEXT NOT NULL,
                info TEXT NOT NULL,
                url TEXT NOT NULL,
                UNIQUE(lineid, info, url)
            )
        ''')
        print("Executed CREATE TABLE IF NOT EXISTS for user_pinned.")
        # 創建 linebot_message 表（若不存在）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS linebot_message (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_id TEXT NOT NULL,
                category TEXT,
                date TEXT,
                event TEXT,
                notes TEXT,
                location TEXT,
                group_id TEXT
            )
        ''')
        print("Executed CREATE TABLE IF NOT EXISTS for linebot_message.")
        conn.commit()
        print("Database initialized and changes committed.")
    except Exception as e:
        print(f"Error initializing database: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            print("Database connection closed after init.")

# 儲存使用者資訊
def save_user(line_id, name):
    print(f"Attempting to save user: line_id={line_id}, name={name}")
    conn = None
    try:
        conn = sqlitecloud.connect(SQLITE_CLOUD_URL)
        cursor = conn.cursor()
        print("Database connection successful for saving user.")
        cursor.execute("INSERT INTO line_users (line_id, name) VALUES (?, ?)", (line_id, name))
        conn.commit()
        print(f"Successfully inserted user: line_id={line_id}")
    except sqlitecloud.IntegrityError:
        print(f"User already exists: line_id={line_id}. Skipping insert.")
    except Exception as e:
        print(f"Error saving user: {e}")
        if conn:
            conn.rollback()
            print("Transaction rolled back due to error.")
    finally:
        if conn:
            conn.close()
            print("Database connection closed after saving user.")

# 取得 LINE 登入 URL
def get_login_url():
    params = {
        "response_type": "code",
        "client_id": LINE_LOGIN_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": "profile openid",
        "state": "random_state_string"
    }
    print(f"Debug - Redirect URI: {REDIRECT_URI}")
    return f"{LINE_AUTH_URL}?{urlencode(params)}"

# 取得 Access Token
def get_access_token(code):
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": LINE_LOGIN_CLIENT_ID,
        "client_secret": LINE_LOGIN_CLIENT_SECRET
    }
    response = requests.post(LINE_TOKEN_URL, data=data)
    return response.json()

# 取得使用者資訊
def get_user_profile(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(LINE_PROFILE_URL, headers=headers)
    return response.json()

# 登入檢查裝飾器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_lineid' not in session:
            return redirect('/login-page')
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    user_info = None
    if 'user_lineid' in session:
        user_info = {
            'line_id': session['user_lineid'],
            'name': session.get('user_name', '使用者')
        }
    return render_template('main.html', user=user_info)

@app.route('/notebook')
@login_required
def notebook():
    user_info = {
        'line_id': session['user_lineid'],
        'name': session.get('user_name', '使用者')
    }
    return render_template('notebook.html', user=user_info)

@app.route('/login-page')
def login_page():
    if 'user_lineid' in session:
        return redirect('/')
    return render_template('login.html')

@app.route('/login')
def login():
    if 'user_lineid' in session:
        return redirect('/')
    if not REDIRECT_URI:
        return "Error: REDIRECT_URI is not set in environment variables", 500
    login_url = get_login_url()
    print(f"Debug - Login URL: {login_url}")
    return redirect(login_url)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    if not code:
        return jsonify({'error': 'No code provided'}), 400

    token_response = get_access_token(code)
    if "access_token" not in token_response:
        return jsonify({'error': 'Failed to get access token'}), 400

    access_token = token_response["access_token"]
    profile = get_user_profile(access_token)
    
    line_id = profile.get("userId")
    name = profile.get("displayName")

    if line_id and name:
        save_user(line_id, name)
        session['user_lineid'] = line_id
        session['user_name'] = name
        return redirect('/')
    
    return jsonify({'error': 'Failed to get user profile'}), 400

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login-page')

@app.route('/api/user')
@login_required
def get_user_info():
    return jsonify({
        'line_id': session['user_lineid'],
        'name': session.get('user_name')
    })

@app.route('/api/messages', methods=['GET'])
@login_required
def get_messages():
    try:
        line_id = session['user_lineid']
        group_id = request.args.get('group_id')
        groups_only = request.args.get('groups_only', 'false').lower() == 'true'

        conn = sqlitecloud.connect(SQLITE_CLOUD_URL)
        cursor = conn.cursor()

        if groups_only:
            cursor.execute("""
                SELECT DISTINCT group_id, group_name 
                FROM linebot_message 
                WHERE line_id = ? AND group_id IS NOT NULL
            """, (line_id,))
            groups = [{'group_id': row[0], 'group_name': row[1]} for row in cursor.fetchall()]
            conn.close()
            return jsonify({'groups': groups}), 200

        if group_id:
            cursor.execute("""
                SELECT category, date, event, notes, location, group_id, group_name
                FROM linebot_message 
                WHERE line_id = ? AND group_id = ?
            """, (line_id, group_id))
        else:
            cursor.execute("""
                SELECT category, date, event, notes, location, group_id, group_name
                FROM linebot_message 
                WHERE line_id = ?
            """, (line_id,))

        messages = [
            {
                'category': row[0],
                'date': row[1],
                'event': row[2],
                'notes': row[3],
                'location': row[4],
                'group_id': row[5],
                'group_name': row[6]
            }
            for row in cursor.fetchall()
        ]
        conn.close()
        return jsonify({'messages': messages}), 200
    except Exception as e:
        print(f"Get messages error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
@login_required
def search():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        print(f"Received search query: {query}")

        if not query:
            return jsonify({'error': '請輸入搜尋內容'}), 400

        print("Generating embedding...")
        query_vector = get_embedding(query)
        print("Embedding generated successfully")

        print("Querying Pinecone index...")
        print("search_index type:", type(search_index), search_index)
        results = search_index.query(vector=query_vector.tolist(), top_k=7, include_metadata=True)
        print("Pinecone query completed")
        
        print("Metadata contents:", [match.get('metadata', {}) for match in results['matches']])

        formatted_results = [
            {
                'id': match['id'],
                'score': match['score'],
                'metadata': match.get('metadata', {})
            }
            for match in results['matches']
        ]

        return jsonify({'results': formatted_results})

    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/pin', methods=['POST'])
@login_required
def pin_result():
    try:
        data = request.get_json()
        info = data.get('info', '').strip()
        url = data.get('url', '').strip()
        lineid = session['user_lineid']

        if not info or not url:
            return jsonify({'error': '缺少必要的資料'}), 400

        conn = sqlitecloud.connect(SQLITE_CLOUD_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM user_pinned WHERE lineid = ? AND info = ? AND url = ?", (lineid, info, url))
        if cursor.fetchone()[0] > 0:
            conn.close()
            return jsonify({'message': '此資料已定選過'}), 200

        cursor.execute("INSERT INTO user_pinned (lineid, info, url) VALUES (?, ?, ?)", (lineid, info, url))
        conn.commit()
        conn.close()
        return jsonify({'message': '已成功儲存搜尋結果'}), 200

    except sqlitecloud.IntegrityError:
        print("Duplicate entry detected, skipping insertion.")
        return jsonify({'message': '此資料已定選過'}), 200
    except Exception as e:
        print(f"Pin error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/unpin', methods=['POST'])
@login_required
def unpin_result():
    try:
        data = request.get_json()
        info = data.get('info', '').strip()
        url = data.get('url', '').strip()
        lineid = session['user_lineid']

        if not info or not url:
            return jsonify({'error': '缺少必要的資料'}), 400

        conn = sqlitecloud.connect(SQLITE_CLOUD_URL)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_pinned WHERE lineid = ? AND info = ? AND url = ?", (lineid, info, url))
        conn.commit()
        conn.close()
        return jsonify({'message': '已成功移除定選資料'}), 200

    except Exception as e:
        print(f"Unpin error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_pinned', methods=['GET'])
@login_required
def get_pinned():
    try:
        lineid = session['user_lineid']
        conn = sqlitecloud.connect(SQLITE_CLOUD_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT info, url FROM user_pinned WHERE lineid = ?", (lineid,))
        pinned_items = cursor.fetchall()
        conn.close()
        
        results = [{'info': item[0], 'url': item[1]} for item in pinned_items]
        return jsonify({'pinned': results}), 200

    except Exception as e:
        print(f"Get pinned error: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/check_pinned', methods=['POST'])
@login_required
def check_pinned():
    try:
        data = request.get_json()
        info = data.get('info', '').strip()
        url = data.get('url', '').strip()
        lineid = session['user_lineid']

        if not info or not url:
            return jsonify({'error': '缺少必要的資料'}), 400

        conn = sqlitecloud.connect(SQLITE_CLOUD_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM user_pinned WHERE lineid = ? AND info = ? AND url = ?", (lineid, info, url))
        exists = cursor.fetchone()[0] > 0
        conn.close()
        return jsonify({'exists': exists}), 200

    except Exception as e:
        print(f"Check pinned error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
