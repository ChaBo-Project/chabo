FROM ghcr.io/chabo-project/hf-chat-ui:0.9.4-chabo

USER root
COPY custom_startup.sh /usr/local/bin/custom_startup.sh
RUN chmod +x /usr/local/bin/custom_startup.sh

WORKDIR /app

USER user
CMD ["/usr/local/bin/custom_startup.sh"]
