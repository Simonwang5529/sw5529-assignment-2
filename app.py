from flask import Flask, jsonify, request
import numpy as np
import matplotlib
import time

matplotlib.use('Agg')  # Non-GUI backend for rendering
from io import BytesIO
import base64
from kmeans import KMeans  # Ensure you have your KMeans implementation in kmeans.py

app = Flask(__name__)

# Store global state of KMeans algorithm
kmeans_state = {
    'dataset': None,
    'centroids': None,
    'labels': None,
    'iterations': 0,
    'n_clusters': 3,
    'is_running': False,
    'manual_init': False,
    'custom_centroids': []
}

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>KMeans Visualization</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            #plot { width: 600px; height: 400px; }
        </style>
    </head>
    <body>
        <h1>KMeans Clustering Visualization</h1>
        <form id="kmeansForm" onsubmit="startKMeans(); return false;">
            <label for="n_clusters">Number of Clusters:</label>
            <input type="number" id="n_clusters" name="n_clusters" value="3" min="1" required>
            <label for="init_method">Initialization Method:</label>
            <select id="init_method" name="init_method">
                <option value="random">Random</option>
                <option value="manual">Manual</option>
            </select>
            <button type="submit">Start KMeans</button>
        </form>
        <button onclick="nextStep()">Next Step</button>
        <button onclick="runToConvergence()">Run to Convergence</button>
        <button onclick="reset()">Reset</button>
        <div id="plot"></div>
        <script>
            let customCentroids = [];

            function startKMeans() {
                const n_clusters = document.getElementById('n_clusters').value;
                const init_method = document.getElementById('init_method').value;

                if (init_method === 'manual') {
                    customCentroids = [];
                    document.getElementById('plot').addEventListener('click', selectCentroid);
                } else {
                    fetch(`/start_kmeans?n_clusters=${n_clusters}&init_method=${init_method}`)
                    .then(response => response.json())
                    .then(data => plotData(data));
                }
            }

            function selectCentroid(event) {
                const rect = event.target.getBoundingClientRect();
                const x = (event.clientX - rect.left) / rect.width;
                const y = 1 - (event.clientY - rect.top) / rect.height;
                customCentroids.push([x, y]);

                if (customCentroids.length >= document.getElementById('n_clusters').value) {
                    document.getElementById('plot').removeEventListener('click', selectCentroid);
                    fetch(`/start_manual_kmeans`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ centroids: customCentroids })
                    })
                    .then(response => response.json())
                    .then(data => plotData(data));
                }
            }

            function nextStep() {
                fetch('/next_step')
                .then(response => response.json())
                .then(data => plotData(data));
            }

            function runToConvergence() {
                fetch('/run_to_convergence')
                .then(response => response.json())
                .then(data => plotData(data));
            }

            function reset() {
                fetch('/reset')
                .then(() => location.reload());
            }

            function plotData(data) {
                var tracePoints = {
                    x: data.points.map(p => p[0]),
                    y: data.points.map(p => p[1]),
                    mode: 'markers',
                    marker: {
                        color: data.labels,
                        size: 10
                    },
                    type: 'scatter'
                };

                var traceCentroids = {
                    x: data.centroids.map(c => c[0]),
                    y: data.centroids.map(c => c[1]),
                    mode: 'markers',
                    marker: {
                        color: 'red',
                        size: 20,
                        symbol: 'x'
                    },
                    type: 'scatter'
                };

                var layout = {
                    title: 'KMeans Clustering',
                    showlegend: false
                };

                Plotly.newPlot('plot', [tracePoints, traceCentroids], layout);
            }
        </script>
    </body>
    </html>
    '''

@app.route('/start_kmeans')
def start_kmeans():
    global kmeans_state
    n_clusters = int(request.args.get('n_clusters', 3))
    init_method = request.args.get('init_method', 'random')

    kmeans_state['n_clusters'] = n_clusters
    X = np.random.rand(300, 2)
    kmeans_state['dataset'] = X

    if init_method == 'random':
        random_idx = np.random.choice(range(len(X)), n_clusters, replace=False)
        centroids = X[random_idx]
        kmeans_state['centroids'] = centroids
        kmeans_state['iterations'] = 0
        kmeans_state['labels'] = np.zeros(len(X))
        kmeans_state['is_running'] = True
        kmeans_state['manual_init'] = False

    return jsonify({
        'points': X.tolist(),
        'centroids': centroids.tolist(),
        'labels': kmeans_state['labels'].tolist(),
        'converged': False
    })

@app.route('/start_manual_kmeans', methods=['POST'])
def start_manual_kmeans():
    global kmeans_state
    X = np.random.rand(300, 2)
    kmeans_state['dataset'] = X

    data = request.json
    kmeans_state['centroids'] = np.array(data['centroids'])
    kmeans_state['manual_init'] = True
    kmeans_state['iterations'] = 0
    kmeans_state['labels'] = np.zeros(len(X))
    kmeans_state['is_running'] = True

    return jsonify({
        'points': X.tolist(),
        'centroids': kmeans_state['centroids'].tolist(),
        'labels': kmeans_state['labels'].tolist(),
        'converged': False
    })

@app.route('/next_step')
def next_step():
    global kmeans_state

    if not kmeans_state['is_running']:
        return jsonify({'converged': True})

    X = kmeans_state['dataset']
    centroids = kmeans_state['centroids']
    n_clusters = kmeans_state['n_clusters']

    # Assign clusters based on current centroids
    labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

    # Update centroids based on new assignments
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])

    # Check for convergence
    converged = np.all(centroids == new_centroids)
    kmeans_state['centroids'] = new_centroids
    kmeans_state['labels'] = labels
    kmeans_state['iterations'] += 1

    if converged or kmeans_state['iterations'] >= 300:
        kmeans_state['is_running'] = False

    # Explicitly convert the 'converged' boolean to an integer to avoid serialization issues
    return jsonify({
        'points': X.tolist(),
        'centroids': new_centroids.tolist(),
        'labels': labels.tolist(),
        'converged': int(converged)  # Convert to an int to avoid bool serialization issues
    })

@app.route('/run_to_convergence')
def run_to_convergence():
    global kmeans_state

    X = kmeans_state['dataset']
    centroids = kmeans_state['centroids']
    n_clusters = kmeans_state['n_clusters']

    labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

    while kmeans_state['is_running']:
        # Assign clusters based on current centroids
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
        
        # Update centroids based on new assignments
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])

        # Check for convergence
        if np.all(centroids == new_centroids) or kmeans_state['iterations'] >= 300:
            kmeans_state['is_running'] = False
            break

        # Update centroids and labels for the next iteration
        centroids = new_centroids
        kmeans_state['centroids'] = centroids
        kmeans_state['labels'] = labels
        kmeans_state['iterations'] += 1

    # After convergence, return the final state
    return jsonify({
        'points': X.tolist(),
        'centroids': centroids.tolist(),
        'labels': labels.tolist(),
        'converged': True
    })

@app.route('/reset')
def reset():
    global kmeans_state
    kmeans_state = {
        'dataset': None,
        'centroids': None,
        'labels': None,
        'iterations': 0,
        'n_clusters': 3,
        'is_running': False,
        'manual_init': False,
        'custom_centroids': []
    }
    return '', 204


if __name__ == '__main__':
    app.run(port=3000, debug=True)