FROM nginx:alpine

WORKDIR /usr/share/nginx/html

# Remove default Nginx static assets
RUN rm -rf ./*

# Copy static files for chat interface
COPY ./static/index.html .
COPY ./static/style.css .
COPY ./static/script.js .
COPY ./static/websocket-test.html .

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
