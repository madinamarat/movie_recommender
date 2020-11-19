from flask import Flask, render_template, request, flash
from cf_recommender import cf_recommender
from nmf_recommender import nmf_recommender
from dictionaries import top50

app = Flask(__name__)  # instantiating a flask application, "__name__" is a reference to the current script

# HTTP request triggers route function, which renders HTML
@app.route('/')
def hello():
    return render_template('index.html')
    # automatically looking for templates/index.html

@app.route('/recommender')
def rec():

    user_input = dict(request.args)
    movie_titles = list(user_input.values())
    recommendations, titles = nmf_recommender(movie_titles)
    similar_movies = cf_recommender(movie_titles)

    return render_template('recommender.html',
                            movie_titles=movie_titles,
                            recommendations = recommendations,
                            similar_movies= similar_movies,
                            titles=titles)

@app.route('/topfifty')
def topfifty():
    top_fifty=top50

    return render_template('topfifty.html',
                            top_fifty=top_fifty)

if __name__ == '__main__':
    app.run(debug=True)