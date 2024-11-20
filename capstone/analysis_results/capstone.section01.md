# Capstone: IT Service Ticket Classification
<b color="red">NOTE: All images are clickable and higher resolution images will load in new browser window</b>
## Background
Most companies have a service desk to help their user or customers with their IT problems.
Customers log and track their ticket in some digital platform and these get routed to Agents to solve.
Routing these tickets to the correct person/team to solve is very important:
- A common approach is to "tag" the ticket with some classification and there is a mapping that routes based on the tag to the correct team to solve it.
- Tickets incorrectly classified result in them "bouncing around" and wasting both agents and customer time.
- There is a financial impact resulting in wasted internal resources and lower customer satisfaction costing the company in real world dollars as well as reputation.
- The scale of the problem grows as you service more users as you need more and more agents to help.


This project aims to address this task by finding and building a model to aid in the classification of these tickets into appropriate categories. We will experiment with various types of pre-processing, vectorizers and regression models to help us find the best one to solve for this.

**Dataset**: [IT Service Ticket Classification Dataset](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)
- This dataset contains 47,837 rows of data and 2 features.

We will follow the CRISP-DM process model consisting of Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment stages.
