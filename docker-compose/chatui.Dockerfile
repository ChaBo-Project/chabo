FROM ghcr.io/m-tyrrell/chat-ui-db:0.9.4-patched

USER root
COPY custom_startup.sh /usr/local/bin/custom_startup.sh
RUN chmod +x /usr/local/bin/custom_startup.sh

WORKDIR /app

USER user
CMD ["/usr/local/bin/custom_startup.sh"]
