FROM ollama/ollama

COPY entrypoint.sh /entrypoint.sh
# Strip CR so a Windows checkout still runs in the Linux image.
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/bin/sh", "/entrypoint.sh"]
