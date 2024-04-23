import streamlit as st
import pickle
import re
import requests

# Load models and vectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(resume_text):
    """Clean the resume text."""
    # Remove URLs
    clean_text = re.sub(r'http\S+\s', ' ', resume_text)
    # Remove @mentions
    clean_text = re.sub(r'@\S+', ' ', clean_text)
    # Remove hashtags
    clean_text = re.sub(r'#\S+\s', ' ', clean_text)
    # Remove RT and cc
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    # Remove special characters
    clean_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', clean_text)
    # Remove non-ASCII characters
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    # Remove extra whitespaces
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text

def fetch_job_listings(query):
    url = "https://jsearch.p.rapidapi.com/search"
    querystring = {
        "query": query,
        "page": "1",
        "num_pages": "1"
    }
    headers = {
        "X-RapidAPI-Key": "16faff3115msh1f4b69f60d2c424p151828jsn464e8c6496be",
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            job_data = response.json()
            if isinstance(job_data, dict) and 'data' in job_data:
                job_listings = job_data['data']
                return job_listings
            else:
                st.error("Failed to fetch job listings. Invalid response format.")
        else:
            st.error(f"Failed to fetch job listings. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload your Resume', type=['pdf', 'txt'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode("latin-1")

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidf_vectorizer.transform([cleaned_resume])
        predicted_category_id = clf.predict(input_features)[0]

        category_mapping = {
            0: "Advocate", 1: "Arts", 2: "Automation Testing", 3: "Blockchain",
            4: "Business Analyst", 5: "Civil Engineer", 6: "Data Science",
            7: "Database", 8: "DevOps Engineer", 9: "DotNet Developer",
            10: "ETL Developer", 11: "Electrical Engineering", 12: "HR",
            13: "Hadoop", 14: "Health and Fitness", 15: "Java Developer",
            16: "Mechanical Engineering", 17: "Network Security Engineer",
            18: "Operations Manager", 19: "PMO", 20: "Python Developer",
            21: "SAP Developer", 22: "Sales", 23: "Testing", 24: "Web Designing",
        }

        predicted_category = category_mapping.get(predicted_category_id, "Unknown")
        st.write("Predicted Category:", predicted_category)

        query = st.text_input("Enter job search query", predicted_category)
        if st.button("Search Jobs"):
            job_listings = fetch_job_listings(query)
            if job_listings:
                st.header("Job Opportunities")
                for job in job_listings:
                    with st.expander(f"{job.get('job_title', 'Title not available')} - {job.get('employer_name', 'Employer not available')}"):
                        st.subheader("Location")
                        st.write(f"{job.get('job_city', 'City not available')}, {job.get('job_state', 'State not available')}")
                        st.subheader("Employment Type")
                        st.write(job.get('job_employment_type', 'Employment type not available'))
                        st.subheader("Posted at")
                        st.write(job.get('job_posted_at_datetime_utc', 'Posted time not available'))

                        st.subheader("Description")
                        st.write(job.get('job_description', 'Description not available'))
                        st.subheader("Apply Link")
                        st.write(job.get('job_apply_link', 'Apply link not available'))
            else:
                st.warning("No job opportunities found for the provided search criteria.")

if __name__ == "__main__":
    main()
